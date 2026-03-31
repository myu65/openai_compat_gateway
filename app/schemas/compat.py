from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any
    name: str | None = None
    tool_call_id: str | None = None


class FunctionSpec(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolSpec(BaseModel):
    type: Literal["function"]
    function: FunctionSpec


class BuiltinToolsConfig(BaseModel):
    web_search: bool | None = None
    file_search: dict[str, Any] | None = None
    code_interpreter: bool | None = None


class ChatCompletionsRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    tools: list[ToolSpec] | None = None
    tool_choice: Any = "auto"
    stream: bool = False
    temperature: float | None = None
    metadata: dict[str, Any] | None = None
    x_builtin_tools: BuiltinToolsConfig | None = None
