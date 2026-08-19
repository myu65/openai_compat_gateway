from __future__ import annotations

import json
from typing import Any

from app.schemas.compat import BuiltinToolsConfig, ToolSpec


def _copy_prompt_cache_breakpoint(source: dict[str, Any], target: dict[str, Any]) -> None:
    breakpoint = source.get("prompt_cache_breakpoint")
    if breakpoint is not None:
        target["prompt_cache_breakpoint"] = breakpoint


def _to_responses_content_part(part: Any) -> dict[str, Any]:
    if not isinstance(part, dict):
        raise ValueError("Responses translation requires Chat Completions content parts to be objects")

    part_type = part.get("type")

    if part_type == "text":
        target = {"type": "input_text", "text": part.get("text", "")}
        _copy_prompt_cache_breakpoint(part, target)
        return target

    if part_type == "image_url":
        image = part.get("image_url")
        if isinstance(image, str):
            image_url = image
            detail = None
        elif isinstance(image, dict):
            image_url = image.get("url")
            detail = image.get("detail")
        else:
            raise ValueError("Chat Completions image_url content must contain an image URL object")

        if not image_url:
            raise ValueError("Chat Completions image_url content must contain image_url.url")

        target = {"type": "input_image", "image_url": image_url}
        if detail is not None:
            target["detail"] = detail
        _copy_prompt_cache_breakpoint(part, target)
        return target

    if part_type == "file":
        file_spec = part.get("file")
        if not isinstance(file_spec, dict):
            raise ValueError("Chat Completions file content must contain a file object")

        target: dict[str, Any] = {"type": "input_file"}
        for key in ("file_data", "file_id", "file_url", "filename", "detail"):
            if file_spec.get(key) is not None:
                target[key] = file_spec[key]
        _copy_prompt_cache_breakpoint(part, target)
        return target

    if part_type == "input_audio":
        raise ValueError(
            "Responses translation does not support Chat Completions input_audio message content; "
            "use x_openai.mode='chat_completions'"
        )

    # Allow callers using the gateway's permissive schema to provide native
    # Responses content parts directly. These are already in the target shape.
    if part_type in {"input_text", "input_image", "input_file"}:
        return dict(part)

    # Chat Completions can replay an assistant refusal as a typed content part,
    # while Responses input messages have no refusal input-part type. Preserve
    # the visible conversational content as text rather than forwarding an
    # invalid Chat-only shape.
    if part_type == "refusal":
        return {"type": "input_text", "text": part.get("refusal", "")}

    raise ValueError(f"Unsupported Chat Completions content part for Responses translation: {part_type!r}")


def _normalize_message_content(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [_to_responses_content_part(part) for part in content]
    if isinstance(content, dict):
        return [_to_responses_content_part(content)]
    return str(content)


def _normalize_tool_output(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [_to_responses_content_part(part) for part in content]
    return json.dumps(content, ensure_ascii=False)


def to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        tool_calls = m.get("tool_calls") or []
        state = m.get("x_openai") or {}
        response_items = state.get("response_items") if isinstance(state, dict) else None

        if role == "assistant" and response_items:
            out.extend(response_items)
            continue

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

        out.append({"role": role, "content": _normalize_message_content(m.get("content"))})
    return out


def to_responses_custom_tools(custom_tools: list[ToolSpec] | None) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for t in custom_tools or []:
        tool = {
            "type": "function",
            "name": t.function.name,
            "description": t.function.description or "",
            "parameters": t.function.parameters or {"type": "object", "properties": {}},
            # Chat Completions function tools are non-strict by default.
            # Responses may attempt strict mode when this is omitted.
            "strict": t.function.strict if t.function.strict is not None else False,
        }
        tools.append(tool)
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
            if isinstance(builtin_cfg.code_interpreter, dict):
                cfg = {"type": "code_interpreter", **builtin_cfg.code_interpreter}
                cfg.setdefault("container", {"type": "auto"})
            else:
                cfg = {"type": "code_interpreter", "container": {"type": "auto"}}
            merged.append(cfg)
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
    # Required for stateless reasoning continuity with store=false.
    include.append("reasoning.encrypted_content")
    return include


def to_responses_text_config(response_format: dict[str, Any] | None, verbosity: str | None) -> dict[str, Any] | None:
    text: dict[str, Any] = {}
    if verbosity is not None:
        text["verbosity"] = verbosity
    if response_format:
        fmt = dict(response_format)
        if fmt.get("type") == "json_schema" and isinstance(fmt.get("json_schema"), dict):
            schema = dict(fmt.pop("json_schema"))
            fmt.update(schema)
        text["format"] = fmt
    return text or None
