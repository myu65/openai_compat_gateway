from __future__ import annotations

import json
from typing import Any


def _collect_message_text(item: Any) -> str:
    parts: list[str] = []
    for c in getattr(item, "content", []) or []:
        ctype = getattr(c, "type", None)
        if ctype in ("output_text", "text"):
            text = getattr(c, "text", None)
            if text:
                parts.append(text)
    return "".join(parts).strip()


def to_legacy_log_steps(resp) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []

    for item in getattr(resp, "output", []) or []:
        item_type = getattr(item, "type", None)

        if item_type == "function_call":
            pending_tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {"name": item.name, "arguments": item.arguments},
                }
            )

        elif item_type == "message":
            steps.append(
                {
                    "role": "assistant",
                    "content": _collect_message_text(item),
                    "tool_calls": pending_tool_calls or None,
                }
            )
            pending_tool_calls = []

        elif item_type == "web_search_call":
            action = getattr(item, "action", None)
            payload = {
                "query": getattr(action, "query", None) if action else None,
                "sources": [
                    {"title": getattr(s, "title", None), "url": getattr(s, "url", None)}
                    for s in (getattr(action, "sources", None) or [])
                ],
            }
            steps.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": getattr(item, "id", "builtin_web_search"),
                            "type": "function",
                            "function": {
                                "name": "openai_builtin.web_search",
                                "arguments": json.dumps({"query": payload["query"]}, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            steps.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(item, "id", "builtin_web_search"),
                    "name": "openai_builtin.web_search",
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            )

        elif item_type == "file_search_call":
            results = [
                {
                    "file_id": getattr(r, "file_id", None),
                    "filename": getattr(r, "filename", None),
                    "score": getattr(r, "score", None),
                    "text": getattr(r, "text", None),
                }
                for r in (getattr(item, "results", None) or [])
            ]
            call_id = getattr(item, "id", "builtin_file_search")
            steps.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "openai_builtin.file_search", "arguments": "{}"},
                        }
                    ],
                }
            )
            steps.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "openai_builtin.file_search",
                    "content": json.dumps({"results": results}, ensure_ascii=False),
                }
            )

        elif item_type == "code_interpreter_call":
            outputs = getattr(item, "outputs", None) or []
            call_id = getattr(item, "id", "builtin_code_interpreter")
            steps.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "openai_builtin.code_interpreter", "arguments": "{}"},
                        }
                    ],
                }
            )
            steps.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "openai_builtin.code_interpreter",
                    "content": json.dumps({"outputs": outputs}, ensure_ascii=False, default=str),
                }
            )

    return steps
