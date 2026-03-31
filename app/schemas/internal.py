from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    url: str | None = None
    title: str | None = None


class BuiltinToolEvent(BaseModel):
    id: str
    type: str
    status: str
    payload: dict[str, Any] | None = None


class BridgeExecution(BaseModel):
    requested_tool_name: str
    builtin_tool_type: str
    execution_mode: str = "gateway_builtin_bridge"
    args: dict[str, Any] = Field(default_factory=dict)
    display_tool_name: str | None = None


class NormalizedResponse(BaseModel):
    assistant_text: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    builtin_tool_events: list[BuiltinToolEvent] = Field(default_factory=list)
    file_search_results: list[dict[str, Any]] = Field(default_factory=list)
    code_interpreter_outputs: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] | None = None
    legacy_steps: list[dict[str, Any]] = Field(default_factory=list)
    bridge_executions: list[BridgeExecution] = Field(default_factory=list)
