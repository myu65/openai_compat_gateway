from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


class ToolExecutor:
    def __init__(self, registry: dict[str, Callable[..., Any]] | None = None):
        self.registry = registry or {}

    def has(self, name: str) -> bool:
        return name in self.registry

    def execute(self, name: str, arguments_json: str) -> str:
        if name not in self.registry:
            raise KeyError(f"Unknown custom tool: {name}")
        fn = self.registry[name]
        args = json.loads(arguments_json or "{}")
        result = fn(**args)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)
