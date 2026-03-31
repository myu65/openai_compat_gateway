from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BuiltinBridgeSpec:
    builtin_tool_type: str
    query_arg_candidates: tuple[str, ...] = ("query", "q", "text")


BUILTIN_TOOL_BRIDGES: dict[str, BuiltinBridgeSpec] = {
    "web_search": BuiltinBridgeSpec("web_search"),
    "search_web": BuiltinBridgeSpec("web_search"),
    "browser_search": BuiltinBridgeSpec("web_search"),
    "file_search": BuiltinBridgeSpec("file_search", query_arg_candidates=("query", "queries", "q", "text")),
}


def find_bridge_for_tool_name(name: str) -> BuiltinBridgeSpec | None:
    return BUILTIN_TOOL_BRIDGES.get(name)


def extract_bridge_query(arguments: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            return " ".join(v for v in value if v.strip()).strip() or None
    return None
