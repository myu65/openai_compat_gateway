from __future__ import annotations

from typing import Any

from app.mappers.legacy_log_mapper import to_legacy_log_steps
from app.schemas.internal import BridgeExecution, BuiltinToolEvent, Citation, NormalizedResponse


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: _jsonable(item) for key, item in vars(value).items() if not key.startswith("_") and item is not None
        }
    return value


def serialize_response_items(resp: Any) -> list[dict[str, Any]]:
    """Serialize output items so a client can replay them with store=false."""

    return [_jsonable(item) for item in (getattr(resp, "output", None) or [])]


def _normalize_usage(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        raw = dict(usage)
    elif hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    elif hasattr(usage, "dict"):
        raw = usage.dict()
    else:
        raw = {key: value for key, value in vars(usage).items() if not key.startswith("_")}

    prompt_tokens = raw.get("prompt_tokens", raw.get("input_tokens", 0))
    completion_tokens = raw.get("completion_tokens", raw.get("output_tokens", 0))
    normalized = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": raw.get("total_tokens", prompt_tokens + completion_tokens),
    }
    prompt_details = raw.get("prompt_tokens_details", raw.get("input_tokens_details"))
    completion_details = raw.get("completion_tokens_details", raw.get("output_tokens_details"))
    if prompt_details is not None:
        normalized["prompt_tokens_details"] = _jsonable(prompt_details)
    if completion_details is not None:
        normalized["completion_tokens_details"] = _jsonable(completion_details)
    return normalized


def _append_annotation_citations(citations: list[Citation], content_item: Any) -> None:
    # Web search answers can attach the user-facing citations to message annotations
    # even when the underlying tool event exposes only opaque/internal sources.
    for annotation in getattr(content_item, "annotations", []) or []:
        if getattr(annotation, "type", None) != "url_citation":
            continue
        citations.append(
            Citation(
                url=getattr(annotation, "url", None),
                title=getattr(annotation, "title", None),
            )
        )


def normalize_final_response(resp, bridge_executions: list[BridgeExecution] | None = None) -> NormalizedResponse:
    assistant_text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    citations: list[Citation] = []
    builtin_tool_events: list[BuiltinToolEvent] = []
    file_search_results: list[dict[str, Any]] = []
    code_interpreter_outputs: list[dict[str, Any]] = []

    for item in getattr(resp, "output", []) or []:
        item_type = getattr(item, "type", None)

        if item_type == "message":
            for c in getattr(item, "content", []) or []:
                ctype = getattr(c, "type", None)
                if ctype in ("output_text", "text"):
                    text = getattr(c, "text", None)
                    if text:
                        assistant_text_parts.append(text)
                    _append_annotation_citations(citations, c)

        elif item_type == "function_call":
            tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {"name": item.name, "arguments": item.arguments},
                }
            )

        elif item_type == "web_search_call":
            action = getattr(item, "action", None)
            payload = {
                "query": getattr(action, "query", None) if action else None,
                "sources_count": len(getattr(action, "sources", None) or []) if action else 0,
            }
            builtin_tool_events.append(
                BuiltinToolEvent(
                    id=getattr(item, "id", "web_search"),
                    type="web_search",
                    status=getattr(item, "status", "completed"),
                    payload=payload,
                )
            )
            if action and getattr(action, "sources", None):
                # Search sources are not guaranteed to be normal web pages. For
                # weather-like queries the tool may emit `type="api"` sources
                # without URL/title, so keep these fields nullable.
                for s in action.sources:
                    citations.append(
                        Citation(
                            url=getattr(s, "url", None),
                            title=getattr(s, "title", None),
                        )
                    )

        elif item_type == "file_search_call":
            results = getattr(item, "results", None) or []
            builtin_tool_events.append(
                BuiltinToolEvent(
                    id=getattr(item, "id", "file_search"),
                    type="file_search",
                    status=getattr(item, "status", "completed"),
                    payload={"results_count": len(results)},
                )
            )
            for r in results:
                file_search_results.append(
                    {
                        "file_id": getattr(r, "file_id", None),
                        "filename": getattr(r, "filename", None),
                        "score": getattr(r, "score", None),
                        "text": getattr(r, "text", None),
                    }
                )

        elif item_type == "code_interpreter_call":
            outputs = getattr(item, "outputs", None) or []
            builtin_tool_events.append(
                BuiltinToolEvent(
                    id=getattr(item, "id", "code_interpreter"),
                    type="code_interpreter",
                    status=getattr(item, "status", "completed"),
                    payload={"outputs_count": len(outputs)},
                )
            )
            for out in outputs:
                code_interpreter_outputs.append(
                    {
                        "type": getattr(out, "type", None),
                        "data": _jsonable(out),
                    }
                )

    return NormalizedResponse(
        assistant_text="".join(assistant_text_parts).strip(),
        tool_calls=tool_calls,
        citations=citations,
        builtin_tool_events=builtin_tool_events,
        file_search_results=file_search_results,
        code_interpreter_outputs=code_interpreter_outputs,
        usage=_normalize_usage(getattr(resp, "usage", None)),
        legacy_steps=to_legacy_log_steps(resp, bridge_executions=bridge_executions),
        bridge_executions=bridge_executions or [],
        response_items=serialize_response_items(resp),
        response_id=getattr(resp, "id", None),
        status=getattr(resp, "status", None),
        incomplete_details=_jsonable(getattr(resp, "incomplete_details", None)),
    )
