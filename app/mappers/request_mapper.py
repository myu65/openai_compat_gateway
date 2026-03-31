from __future__ import annotations

import json
from typing import Any

from app.schemas.compat import BuiltinToolsConfig, ToolSpec


def _normalize_message_content(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, (str, list)):
        return content
    if isinstance(content, dict):
        return content
    return str(content)


def _normalize_tool_output(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        tool_calls = m.get("tool_calls") or []

        if role == "assistant" and tool_calls:
            content = _normalize_message_content(m.get("content"))
            if content not in ("", [], None):
                out.append({"role": "assistant", "content": content})

            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                out.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.get("id"),
                        "name": function.get("name"),
                        "arguments": function.get("arguments", "{}"),
                    }
                )
            continue

        if role == "tool":
            out.append(
                {
                    "type": "function_call_output",
                    "call_id": m.get("tool_call_id"),
                    "output": _normalize_tool_output(m.get("content")),
                }
            )
            continue

        item: dict[str, Any] = {"role": role, "content": _normalize_message_content(m.get("content"))}
        if m.get("name"):
            item["name"] = m["name"]
        if m.get("tool_call_id"):
            item["tool_call_id"] = m["tool_call_id"]
        out.append(item)
    return out


def to_responses_custom_tools(custom_tools: list[ToolSpec] | None) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for t in custom_tools or []:
        tools.append(
            {
                "type": "function",
                "name": t.function.name,
                "description": t.function.description or "",
                "parameters": t.function.parameters or {"type": "object", "properties": {}},
            }
        )
    return tools


def merge_builtin_tools(tools: list[dict[str, Any]], builtin_cfg: BuiltinToolsConfig | None) -> list[dict[str, Any]]:
    merged = list(tools)
    if builtin_cfg:
        if builtin_cfg.web_search:
            merged.append({"type": "web_search"})
        if builtin_cfg.file_search:
            cfg = {"type": "file_search"}
            cfg.update(builtin_cfg.file_search)
            merged.append(cfg)
        if builtin_cfg.code_interpreter:
            merged.append({"type": "code_interpreter"})
    return merged


def build_include_list(
    builtin_cfg: BuiltinToolsConfig | None,
    *,
    include_web_search_results: bool = False,
) -> list[str]:
    include: list[str] = []
    if builtin_cfg and builtin_cfg.web_search:
        include.append("web_search_call.action.sources")
        if include_web_search_results:
            include.append("web_search_call.results")
    if builtin_cfg and builtin_cfg.file_search:
        include.append("file_search_call.results")
    if builtin_cfg and builtin_cfg.code_interpreter:
        include.append("code_interpreter_call.outputs")
    return include
