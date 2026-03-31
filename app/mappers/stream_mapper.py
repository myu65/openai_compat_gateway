from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterable


def _sse_line(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\\n\\n"


def map_stream_events(openai_stream: Iterable[Any], model: str):
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    citations: list[dict[str, Any]] = []
    builtin_tool_events: list[dict[str, Any]] = []
    sent_role = False

    for event in openai_stream:
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

        elif etype == "response.function_call_arguments.delta":
            call_id = getattr(event, "call_id", None)
            delta = getattr(event, "delta", "")
            name = getattr(event, "name", None)
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
                                        "index": 0,
                                        "id": call_id,
                                        "type": "function",
                                        "function": {"name": name, "arguments": delta},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )

        elif etype == "response.output_item.done":
            item = getattr(event, "item", None)
            item_type = getattr(item, "type", None) if item else None
            if item_type == "web_search_call":
                action = getattr(item, "action", None)
                builtin_tool_events.append({"type": "web_search", "status": getattr(item, "status", "completed")})
                if action and getattr(action, "sources", None):
                    for s in action.sources:
                        citations.append({"url": getattr(s, "url", None), "title": getattr(s, "title", None)})
            elif item_type == "file_search_call":
                builtin_tool_events.append({"type": "file_search", "status": getattr(item, "status", "completed")})
            elif item_type == "code_interpreter_call":
                builtin_tool_events.append({"type": "code_interpreter", "status": getattr(item, "status", "completed")})

        elif etype == "response.completed":
            yield _sse_line(
                {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "x_openai": {"citations": citations, "builtin_tool_events": builtin_tool_events},
                }
            )
            yield "data: [DONE]\\n\\n"
            return
