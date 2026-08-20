from __future__ import annotations

import asyncio
import inspect
import json
import logging
import secrets
import threading
import time
import uuid
from collections.abc import AsyncIterable, Iterable
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIConnectionError, APITimeoutError

from app.adapters.client import create_async_openai_client
from app.adapters.openai_chat_completions import AsyncOpenAIChatCompletionsAdapter
from app.adapters.openai_responses import AsyncOpenAIResponsesAdapter
from app.config import settings
from app.logging.audit import AuditLogger
from app.schemas.compat import ChatCompletionsRequest
from app.services.async_chat_service import AsyncChatService
from app.tools.executor import ToolExecutor

router = APIRouter()
logger = logging.getLogger("gateway.runtime")


class InflightLimiter:
    """Process-local non-blocking admission control for long-lived LLM requests."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._active = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._active >= self.capacity:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1

    @property
    def active(self) -> int:
        with self._lock:
            return self._active


_inflight = InflightLimiter(settings.gateway_max_inflight_requests)


def runtime_status() -> dict[str, Any]:
    return {
        "active_requests": _inflight.active,
        "max_inflight_requests": _inflight.capacity,
        "service_initialized": _service is not None,
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


_service: AsyncChatService | None = None
_openai_client = None
_service_init_lock = threading.Lock()


def _openai_client_kwargs() -> dict[str, Any]:
    return {
        "api_key": settings.openai_api_key,
        "connect_timeout_seconds": settings.openai_connect_timeout_seconds,
        "read_timeout_seconds": settings.openai_read_timeout_seconds,
        "write_timeout_seconds": settings.openai_write_timeout_seconds,
        "pool_timeout_seconds": settings.openai_pool_timeout_seconds,
        "max_retries": settings.openai_max_retries,
        "max_connections": settings.openai_max_connections,
        "max_keepalive_connections": settings.openai_max_keepalive_connections,
    }


async def get_service() -> AsyncChatService:
    global _openai_client, _service
    if _service is not None:
        return _service

    with _service_init_lock:
        if _service is None:
            _openai_client = create_async_openai_client(**_openai_client_kwargs())
            adapter = AsyncOpenAIResponsesAdapter(_openai_client)
            native_adapter = AsyncOpenAIChatCompletionsAdapter(_openai_client)
            executor = ToolExecutor(_sample_tool_registry())
            audit = AuditLogger()
            _service = AsyncChatService(
                adapter,
                executor,
                audit,
                default_model=settings.openai_model_default,
                include_web_search_results=settings.openai_include_web_search_results,
                native_adapter=native_adapter,
            )
    return _service


async def close_service() -> None:
    global _openai_client, _service
    client = _openai_client
    _openai_client = None
    _service = None
    if client is not None:
        await client.close()


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
    """Legacy sync mapper retained for dependency-injected sync tests."""
    for chunk in stream:
        yield f"data: {json.dumps(_model_dump(chunk), ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


async def _close_async_iterable(stream: Any) -> None:
    close = getattr(stream, "aclose", None)
    if close is None:
        close = getattr(stream, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def _native_async_sse(stream):
    try:
        async for chunk in stream:
            yield f"data: {json.dumps(_model_dump(chunk), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        await _close_async_iterable(stream)


def _upstream_timeout_error() -> dict[str, Any]:
    return {
        "message": "Upstream OpenAI request timed out",
        "type": "api_error",
        "param": None,
        "code": "upstream_timeout",
    }


def _upstream_connection_error() -> dict[str, Any]:
    return {
        "message": "Upstream OpenAI connection failed",
        "type": "api_error",
        "param": None,
        "code": "upstream_connection_error",
    }


def _gateway_busy_error() -> dict[str, Any]:
    return {
        "message": "Gateway is at its in-flight request limit",
        "type": "api_error",
        "param": None,
        "code": "gateway_busy",
    }


def _internal_stream_error() -> dict[str, Any]:
    return {
        "message": "Internal gateway streaming error",
        "type": "api_error",
        "param": None,
        "code": "gateway_stream_error",
    }


def _timeout_safe_sse(stream: Iterable[str]):
    """Legacy sync timeout wrapper retained for existing tests."""
    try:
        yield from stream
    except APITimeoutError:
        yield f"data: {json.dumps({'error': _upstream_timeout_error()}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


async def _guarded_async_sse(
    stream: AsyncIterable[str],
    *,
    request_id: str,
    mode: str,
    deadline_seconds: float | None = None,
):
    started = time.monotonic()
    outcome = "completed"
    deadline = deadline_seconds or settings.openai_request_deadline_seconds
    try:
        async with asyncio.timeout(deadline):
            async for chunk in stream:
                yield chunk
    except APITimeoutError:
        outcome = "upstream_timeout"
        yield f"data: {json.dumps({'error': _upstream_timeout_error()}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except TimeoutError:
        outcome = "request_deadline"
        yield f"data: {json.dumps({'error': _upstream_timeout_error()}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except APIConnectionError:
        outcome = "upstream_connection_error"
        yield f"data: {json.dumps({'error': _upstream_connection_error()}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        outcome = "client_disconnect"
        raise
    except Exception:
        outcome = "stream_error"
        logger.exception("stream_failed request_id=%s mode=%s", request_id, mode)
        yield f"data: {json.dumps({'error': _internal_stream_error()}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        await _close_async_iterable(stream)
        _inflight.release()
        logger.info(
            "request_finished request_id=%s mode=%s stream=true outcome=%s elapsed_ms=%d active=%d",
            request_id,
            mode,
            outcome,
            int((time.monotonic() - started) * 1000),
            _inflight.active,
        )


def _error_response(exc: Exception, *, request_id: str | None = None) -> JSONResponse:
    headers = {"X-Request-Id": request_id} if request_id else None
    if isinstance(exc, (APITimeoutError, TimeoutError)):
        return JSONResponse({"error": _upstream_timeout_error()}, status_code=504, headers=headers)
    if isinstance(exc, APIConnectionError):
        return JSONResponse({"error": _upstream_connection_error()}, status_code=502, headers=headers)

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
    return JSONResponse({"error": error}, status_code=status_code, headers=headers)


def _sse_headers(request_id: str) -> dict[str, str]:
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "X-Request-Id": request_id,
    }
    if settings.gateway_stream_connection_close:
        # Work around openai-python #3440 for downstream SDK clients until fixed upstream.
        headers["Connection"] = "close"
    return headers


async def _call_async_or_sync(svc, async_name: str, sync_name: str, req):
    async_method = getattr(svc, async_name, None)
    if async_method is not None:
        return await async_method(req)
    return await asyncio.to_thread(getattr(svc, sync_name), req)


@router.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionsRequest,
    svc: AsyncChatService = Depends(get_service),
    _auth: None = Depends(verify_gateway_key),
):
    request_id = uuid.uuid4().hex
    if not _inflight.try_acquire():
        return JSONResponse(
            {"error": _gateway_busy_error()},
            status_code=503,
            headers={"X-Request-Id": request_id, "Retry-After": "1"},
        )

    started = time.monotonic()
    mode = "unknown"
    stream_handoff = False
    try:
        mode = svc.select_mode(req)
        logger.info(
            "request_started request_id=%s mode=%s stream=%s model=%s active=%d",
            request_id,
            mode,
            req.stream,
            req.model or settings.openai_model_default,
            _inflight.active,
        )

        if mode == "chat_completions":
            if req.stream:
                if hasattr(svc, "run_native_stream_async"):
                    upstream = await svc.run_native_stream_async(req)
                    stream = _native_async_sse(upstream)
                    stream_handoff = True
                    return StreamingResponse(
                        _guarded_async_sse(stream, request_id=request_id, mode=mode),
                        media_type="text/event-stream",
                        headers=_sse_headers(request_id),
                    )

                legacy_stream = svc.run_native_stream(req)
                _inflight.release()
                stream_handoff = True
                return StreamingResponse(
                    _timeout_safe_sse(_native_sse(legacy_stream)),
                    media_type="text/event-stream",
                    headers=_sse_headers(request_id),
                )

            async with asyncio.timeout(settings.openai_request_deadline_seconds):
                native = _model_dump(
                    await _call_async_or_sync(svc, "run_native_nonstream_async", "run_native_nonstream", req)
                )
            native["x_openai"] = {"mode": "chat_completions", "store": False}
            return JSONResponse(native, headers={"X-Request-Id": request_id})

        if req.stream:
            if hasattr(svc, "run_stream_async"):
                stream = await svc.run_stream_async(req)
                stream_handoff = True
                return StreamingResponse(
                    _guarded_async_sse(stream, request_id=request_id, mode=mode),
                    media_type="text/event-stream",
                    headers=_sse_headers(request_id),
                )

            legacy_stream = svc.run_stream(req)
            _inflight.release()
            stream_handoff = True
            return StreamingResponse(
                _timeout_safe_sse(legacy_stream),
                media_type="text/event-stream",
                headers=_sse_headers(request_id),
            )

        async with asyncio.timeout(settings.openai_request_deadline_seconds):
            normalized = await _call_async_or_sync(svc, "run_nonstream_async", "run_nonstream", req)
    except Exception as exc:
        return _error_response(exc, request_id=request_id)
    finally:
        if not stream_handoff:
            _inflight.release()
            logger.info(
                "request_finished request_id=%s mode=%s stream=false elapsed_ms=%d active=%d",
                request_id,
                mode,
                int((time.monotonic() - started) * 1000),
                _inflight.active,
            )

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
    return JSONResponse(payload, headers={"X-Request-Id": request_id})
