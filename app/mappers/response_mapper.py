from __future__ import annotations

from typing import Any

from app.mappers.legacy_log_mapper import to_legacy_log_steps
from app.schemas.internal import BridgeExecution, BuiltinToolEvent, Citation, NormalizedResponse


def _normalize_usage(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    return {
        key: value
        for key, value in vars(usage).items()
        if not key.startswith("_")
    }


def _append_annotation_citations(citations: list[Citation], content_item: Any) -> None:
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
                        "data": out,
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
        legacy_steps=to_legacy_log_steps(resp),
        bridge_executions=bridge_executions or [],
    )
