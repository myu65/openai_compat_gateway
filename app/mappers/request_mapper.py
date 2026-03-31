from __future__ import annotations

from typing import Any

from app.schemas.compat import BuiltinToolsConfig, ToolSpec


def to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m["content"]
        if not isinstance(content, str):
            content = str(content)
        item: dict[str, Any] = {"role": m["role"], "content": content}
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


def build_include_list(builtin_cfg: BuiltinToolsConfig | None) -> list[str]:
    include: list[str] = []
    if builtin_cfg and builtin_cfg.web_search:
        include.append("web_search_call.action.sources")
        include.append("web_search_call.results")
    if builtin_cfg and builtin_cfg.file_search:
        include.append("file_search_call.results")
    if builtin_cfg and builtin_cfg.code_interpreter:
        include.append("code_interpreter_call.outputs")
    return include
