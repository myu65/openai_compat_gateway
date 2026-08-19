from __future__ import annotations

import json
import secrets
import time
import uuid
from collections.abc import Iterable
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APITimeoutError

from app.adapters.openai_chat_completions import OpenAIChatCompletionsAdapter
from app.adapters.openai_responses import OpenAIResponsesAdapter
from app.config import settings
from app.logging.audit import AuditLogger
from app.schemas.compat import ChatCompletionsRequest
from app.services.chat_service import ChatService
from app.tools.executor import ToolExecutor

router = APIRouter()
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def verify_gateway_key(authorization: str | None = Header(default=None)) -> None:
    expected = settings.gateway_api_key
    if not expected:
        return
    prefix = "Bearer "
    supplied = authorization[len(prefix) :] if authorization and authorization.startswith(prefix) else ""
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid gateway API key")


def _sample_tool_registry():
    def echo_tool(text: str) -> dict:
        return {"echo": text}

    return {"echo_tool": echo_tool}


_service: ChatService | None = None


def _openai_adapter_kwargs() -> dict[str, Any]:
    return {
        "api_key": settings.openai_api_key,
        "connect_timeout_seconds": settings.openai_connect_timeout_seconds,
        "read_timeout_seconds": settings.openai_read_timeout_seconds,
        "write_timeout_seconds": settings.openai_write_timeout_seconds,
        "pool_timeout_seconds": settings.openai_pool_timeout_seconds,
        "max_retries": settings.openai_max_retries,
    }


def get_service() -> ChatService:
    global _service
    if _service is None:
        adapter = OpenAIResponsesAdapter(**_openai_adapter_kwargs())
        native_adapter = OpenAIChatCompletionsAdapter(**_openai_adapter_kwargs())
        executor = ToolExecutor(_sample_tool_registry())
        audit = AuditLogger()
        _service = ChatService(
            adapter,
            executor,
            audit,
            default_model=settings.openai_model_default,
            include_web_search_results=settings.openai_include_web_search_results,
            native_adapter=native_adapter,
        )
    return _service


def _finish_reason_for_normalized_response(normalized) -> str:
    if normalized.tool_calls:
        return "tool_calls"
    if normalized.incomplete_details and normalized.incomplete_details.get("reason") == "max_output_tokens":
        return "length"
    return "stop"


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    raise TypeError(f"Unsupported OpenAI response type: {type(value)!r}")


def _native_sse(stream: Iterable[Any]):
    for chunk in stream:
        yield f"data: {json.dumps(_model_dump(chunk), ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def _upstream_timeout_error() -> dict[str, Any]:
    return {
        "message": "Upstream OpenAI request timed out",
        "type": "api_error",
        "param": None,
        "code": "upstream_timeout",
    }


def _timeout_safe_sse(stream: Iterable[str]):
    """Finish an already-started SSE response cleanly if the upstream read times out."""
    try:
        yield from stream
    except APITimeoutError:
        yield f"data: {json.dumps({'error': _upstream_timeout_error()}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


def _error_response(exc: Exception) -> JSONResponse:
    if isinstance(exc, APITimeoutError):
        return JSONResponse({"error": _upstream_timeout_error()}, status_code=504)

    default_status = 400 if isinstance(exc, ValueError) else 500
    status_code = int(getattr(exc, "status_code", default_status) or default_status)
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        error = body["error"]
    elif isinstance(body, dict) and body.get("message"):
        error = body
    else:
        error = {
            "message": str(exc) if status_code < 500 else "Internal gateway error",
            "type": "invalid_request_error" if status_code < 500 else "api_error",
            "param": None,
            "code": getattr(exc, "code", None),
        }
    return JSONResponse({"error": error}, status_code=status_code)


@router.post("/v1/chat/completions")
def chat_completions(
    req: ChatCompletionsRequest,
    svc: ChatService = Depends(get_service),
    _auth: None = Depends(verify_gateway_key),
):
    try:
        mode = svc.select_mode(req)
        if mode == "chat_completions":
            if req.stream:
                return StreamingResponse(
                    _timeout_safe_sse(_native_sse(svc.run_native_stream(req))),
                    media_type="text/event-stream",
                    headers=SSE_HEADERS,
                )
            native = _model_dump(svc.run_native_nonstream(req))
            native["x_openai"] = {"mode": "chat_completions", "store": False}
            return JSONResponse(native)

        if req.stream:
            stream = svc.run_stream(req)
            return StreamingResponse(
                _timeout_safe_sse(stream),
                media_type="text/event-stream",
                headers=SSE_HEADERS,
            )

        normalized = svc.run_nonstream(req)
    except Exception as exc:
        return _error_response(exc)

    message_content = None if normalized.tool_calls and not normalized.assistant_text else normalized.assistant_text
    state = {
        "response_items": normalized.response_items,
        "response_id": normalized.response_id,
        "status": normalized.status,
        "incomplete_details": normalized.incomplete_details,
    }
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or settings.openai_model_default,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": message_content,
                    "tool_calls": normalized.tool_calls or None,
                    "x_openai": state,
                },
                "finish_reason": _finish_reason_for_normalized_response(normalized),
            }
        ],
        "usage": normalized.usage,
        "x_openai": {
            "citations": [c.model_dump() for c in normalized.citations],
            "builtin_tool_events": [e.model_dump() for e in normalized.builtin_tool_events],
            "file_search_results": normalized.file_search_results,
            "code_interpreter_outputs": normalized.code_interpreter_outputs,
            "legacy_steps": normalized.legacy_steps,
            "bridge_executions": [b.model_dump() for b in normalized.bridge_executions],
            "mode": "responses",
            **state,
        },
    }
    return JSONResponse(payload)
