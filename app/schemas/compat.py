from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OpenAIStateEnvelope(BaseModel):
    """Lossless Responses state carried through the Chat Completions envelope."""

    model_config = ConfigDict(extra="allow")

    response_items: list[dict[str, Any]] = Field(default_factory=list)


class OpenAICompatConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: Literal["auto", "responses", "chat_completions"] = "auto"
    input_items: list[dict[str, Any]] = Field(default_factory=list)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["developer", "system", "user", "assistant", "tool", "function"]
    content: Any = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    x_openai: OpenAIStateEnvelope | None = None


class FunctionSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool | None = None


class ToolSpec(BaseModel):
    type: Literal["function"]
    function: FunctionSpec


class BuiltinToolsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    web_search: bool | None = None
    file_search: dict[str, Any] | None = None
    code_interpreter: bool | dict[str, Any] | None = None


class ChatCompletionsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage]
    tools: list[ToolSpec] | None = None
    tool_choice: Any = "auto"
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_completion_tokens: int | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    response_format: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    service_tier: str | None = None
    verbosity: str | None = None
    n: int = 1
    stop: str | list[str] | None = None
    metadata: dict[str, Any] | None = None
    x_builtin_tools: BuiltinToolsConfig | None = None
    x_openai: OpenAICompatConfig | None = None
