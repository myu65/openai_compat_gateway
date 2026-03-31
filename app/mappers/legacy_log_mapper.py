from __future__ import annotations

import json
from typing import Any

from app.schemas.internal import BridgeExecution


def _bridge_name_map(bridge_executions: list[BridgeExecution] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for bridge in bridge_executions or []:
        if bridge.display_tool_name and bridge.builtin_tool_type not in mapping:
            mapping[bridge.builtin_tool_type] = bridge.display_tool_name
    return mapping


def _collect_message_text(item: Any) -> str:
    parts: list[str] = []
    for c in getattr(item, "content", []) or []:
        ctype = getattr(c, "type", None)
        if ctype in ("output_text", "text"):
            text = getattr(c, "text", None)
            if text:
                parts.append(text)
    return "".join(parts).strip()


def _collect_url_citations(item: Any) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for content_item in getattr(item, "content", []) or []:
        # The model's final answer can carry richer URL citations here than the
        # raw web_search_call sources, so surface them separately in legacy logs.
        for annotation in getattr(content_item, "annotations", []) or []:
            if getattr(annotation, "type", None) != "url_citation":
                continue
            citations.append(
                {
                    "title": getattr(annotation, "title", None),
                    "url": getattr(annotation, "url", None),
                }
            )
    return citations


def to_legacy_log_steps(resp, bridge_executions: list[BridgeExecution] | None = None) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []
    bridge_names = _bridge_name_map(bridge_executions)

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
                    "citations": _collect_url_citations(item) or None,
                }
            )
            pending_tool_calls = []

        elif item_type == "web_search_call":
            action = getattr(item, "action", None)
            tool_name = bridge_names.get("web_search", "openai_builtin.web_search")
            call_id = getattr(item, "id", "builtin_web_search")
            payload = {
                "query": getattr(action, "query", None) if action else None,
                "sources": [
                    # Preserve the source type from OpenAI because it is not
                    # always a normal URL-backed web page.
                    {
                        "type": getattr(s, "type", None),
                        "title": getattr(s, "title", None),
                        "url": getattr(s, "url", None),
                    }
                    for s in (getattr(action, "sources", None) or [])
                ],
            }
            steps.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps({"query": payload["query"]}, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            steps.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            )

        elif item_type == "file_search_call":
            tool_name = bridge_names.get("file_search", "openai_builtin.file_search")
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
                            "function": {"name": tool_name, "arguments": "{}"},
                        }
                    ],
                }
            )
            steps.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
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
