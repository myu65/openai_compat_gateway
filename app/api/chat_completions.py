from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import settings
from app.logging.audit import AuditLogger
from app.adapters.openai_responses import OpenAIResponsesAdapter
from app.schemas.compat import ChatCompletionsRequest
from app.services.chat_service import ChatService
from app.tools.executor import ToolExecutor

router = APIRouter()


def _sample_tool_registry():
    def echo_tool(text: str) -> dict:
        return {"echo": text}

    return {"echo_tool": echo_tool}


_service: ChatService | None = None


def get_service() -> ChatService:
    global _service
    if _service is None:
        adapter = OpenAIResponsesAdapter(api_key=settings.openai_api_key)
        executor = ToolExecutor(_sample_tool_registry())
        audit = AuditLogger()
        _service = ChatService(
            adapter,
            executor,
            audit,
            default_model=settings.openai_model_default,
            include_web_search_results=settings.openai_include_web_search_results,
        )
    return _service


def _finish_reason_for_normalized_response(normalized) -> str:
    if normalized.tool_calls:
        return "tool_calls"
    return "stop"


@router.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionsRequest, svc: ChatService = Depends(get_service)):
    if req.stream:
        stream = svc.run_stream(req)
        return StreamingResponse(stream, media_type="text/event-stream")

    normalized = svc.run_nonstream(req)
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
                    "content": normalized.assistant_text,
                    "tool_calls": normalized.tool_calls,
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
        },
    }
    return JSONResponse(payload)
