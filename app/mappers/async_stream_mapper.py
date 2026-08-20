from __future__ import annotations

import inspect
import time
import uuid
from types import SimpleNamespace
from typing import Any

from app.mappers.legacy_log_mapper import to_legacy_log_steps
from app.mappers.response_mapper import _jsonable, _normalize_usage, normalize_final_response
from app.mappers.stream_mapper import _collect_message_text, _sse_line, _stream_finish_reason
from app.schemas.internal import BridgeExecution


async def _close_async_stream(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if close is None:
        close = getattr(stream, "aclose", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _last_assistant_text(completed_items: list[Any]) -> str:
    for item in reversed(completed_items):
        if getattr(item, "type", None) == "message":
            return _collect_message_text(item)
    return ""


async def map_async_stream_events(
    openai_stream,
    model: str,
    bridge_executions: list[BridgeExecution] | None = None,
    include_usage: bool = False,
):
    """Map an AsyncOpenAI Responses stream to Chat Completions SSE and always release it."""
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    citations: list[dict[str, Any]] = []
    builtin_tool_events: list[dict[str, Any]] = []
    completed_items: list[Any] = []
    sent_role = False
    call_indexes: dict[str, int] = {}
    announced_calls: set[str | None] = set()
    function_items: dict[str, tuple[str | None, str | None]] = {}

    def call_index(call_id: str | None) -> int:
        key = call_id or f"unknown-{len(call_indexes)}"
        if key not in call_indexes:
            call_indexes[key] = len(call_indexes)
        return call_indexes[key]

    try:
        async for event in openai_stream:
            etype = getattr(event, "type", None)

            if not sent_role:
                sent_role = True
                yield _sse_line(
                    {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                    }
                )

            if etype in ("response.output_text.delta", "response.content_part.delta"):
                delta = getattr(event, "delta", None)
                if delta:
                    yield _sse_line(
                        {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }
                    )

            elif etype == "response.refusal.delta":
                delta = getattr(event, "delta", None)
                if delta:
                    yield _sse_line(
                        {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"refusal": delta}, "finish_reason": None}],
                        }
                    )

            elif etype == "response.output_item.added":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "function_call":
                    item_id = getattr(item, "id", None)
                    call_id = getattr(item, "call_id", None)
                    name = getattr(item, "name", None)
                    if item_id:
                        function_items[item_id] = (call_id, name)
                    index = call_index(call_id)
                    announced_calls.add(call_id)
                    yield _sse_line(
                        {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": index,
                                                "id": call_id,
                                                "type": "function",
                                                "function": {"name": name, "arguments": ""},
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )

            elif etype == "response.function_call_arguments.delta":
                item_id = getattr(event, "item_id", None)
                known_call_id, known_name = function_items.get(item_id, (None, None))
                call_id = getattr(event, "call_id", None) or known_call_id or item_id
                delta = getattr(event, "delta", "")
                name = getattr(event, "name", None) or known_name
                index = call_index(call_id)
                function: dict[str, Any] = {"arguments": delta}
                tool_call: dict[str, Any] = {"index": index, "function": function}
                if call_id not in announced_calls:
                    announced_calls.add(call_id)
                    tool_call.update({"id": call_id, "type": "function"})
                    function["name"] = name
                yield _sse_line(
                    {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": [tool_call]},
                                "finish_reason": None,
                            }
                        ],
                    }
                )

            elif etype == "response.output_item.done":
                item = getattr(event, "item", None)
                item_type = getattr(item, "type", None) if item else None
                if item is not None:
                    completed_items.append(item)
                if item_type == "web_search_call":
                    action = getattr(item, "action", None)
                    builtin_tool_events.append(
                        {
                            "id": getattr(item, "id", "web_search"),
                            "type": "web_search",
                            "status": getattr(item, "status", "completed"),
                            "payload": {
                                "query": getattr(action, "query", None) if action else None,
                                "sources_count": len(getattr(action, "sources", None) or []) if action else 0,
                            },
                        }
                    )
                    if action and getattr(action, "sources", None):
                        for source in action.sources:
                            citations.append(
                                {"url": getattr(source, "url", None), "title": getattr(source, "title", None)}
                            )
                elif item_type == "file_search_call":
                    builtin_tool_events.append(
                        {
                            "id": getattr(item, "id", "file_search"),
                            "type": "file_search",
                            "status": getattr(item, "status", "completed"),
                        }
                    )
                elif item_type == "code_interpreter_call":
                    builtin_tool_events.append(
                        {
                            "id": getattr(item, "id", "code_interpreter"),
                            "type": "code_interpreter",
                            "status": getattr(item, "status", "completed"),
                        }
                    )
                elif item_type == "message":
                    for content_item in getattr(item, "content", []) or []:
                        for annotation in getattr(content_item, "annotations", []) or []:
                            if getattr(annotation, "type", None) != "url_citation":
                                continue
                            citations.append(
                                {
                                    "url": getattr(annotation, "url", None),
                                    "title": getattr(annotation, "title", None),
                                }
                            )

            elif etype in ("response.completed", "response.incomplete"):
                response = getattr(event, "response", None)
                if response is not None and getattr(response, "output", None):
                    completed_items = list(response.output)
                normalized = normalize_final_response(SimpleNamespace(output=completed_items))
                legacy_steps = to_legacy_log_steps(
                    SimpleNamespace(output=completed_items),
                    bridge_executions=bridge_executions,
                )
                state = {
                    "response_items": [_jsonable(item) for item in completed_items],
                    "response_id": getattr(response, "id", None),
                    "status": getattr(response, "status", "completed") if response is not None else "completed",
                    "incomplete_details": _jsonable(getattr(response, "incomplete_details", None)),
                }
                finish_reason = _stream_finish_reason(completed_items)
                incomplete = state["incomplete_details"] or {}
                if isinstance(incomplete, dict) and incomplete.get("reason") == "max_output_tokens":
                    finish_reason = "length"
                yield _sse_line(
                    {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"x_openai": state},
                                "finish_reason": finish_reason,
                            }
                        ],
                        "x_openai": {
                            "citations": citations,
                            "builtin_tool_events": builtin_tool_events,
                            "file_search_results": normalized.file_search_results,
                            "code_interpreter_outputs": normalized.code_interpreter_outputs,
                            "legacy_steps": legacy_steps,
                            "assistant_text": _last_assistant_text(completed_items),
                            **state,
                        },
                    }
                )
                if include_usage:
                    yield _sse_line(
                        {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": _normalize_usage(getattr(response, "usage", None)),
                        }
                    )
                yield "data: [DONE]\n\n"
                return

            elif etype in ("response.failed", "error"):
                response = getattr(event, "response", None)
                error = getattr(event, "error", None) or getattr(response, "error", None)
                yield _sse_line(
                    {
                        "error": {
                            "message": getattr(error, "message", None) or str(error or "Responses stream failed"),
                            "type": getattr(error, "type", None) or "api_error",
                            "code": getattr(error, "code", None),
                        }
                    }
                )
                yield "data: [DONE]\n\n"
                return
    finally:
        await _close_async_stream(openai_stream)
