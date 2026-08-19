from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.schemas.compat import BuiltinToolsConfig, ChatCompletionsRequest, ChatMessage, FunctionSpec, ToolSpec
from app.services.chat_service import ChatService


class CapturingAdapter:
    def __init__(self) -> None:
        self.calls = []

    def create_response(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output=[], usage={"total_tokens": 1})


class NullLogger:
    def log_chat(self, req, normalized):
        return None


class NullExecutor:
    pass


def service(adapter: CapturingAdapter | None = None) -> ChatService:
    return ChatService(
        adapter=adapter or CapturingAdapter(),
        tool_executor=NullExecutor(),
        audit_logger=NullLogger(),
        default_model="gpt-5.6-luna",
    )


def test_unknown_builtin_tool_option_fails_instead_of_being_dropped() -> None:
    req = ChatCompletionsRequest(
        messages=[ChatMessage(role="user", content="hello")],
        x_builtin_tools=BuiltinToolsConfig(web_search=True, future_builtin={"enabled": True}),
    )

    with pytest.raises(ValueError, match="future_builtin"):
        service().run_nonstream(req)


def test_unknown_function_tool_field_fails_instead_of_being_dropped() -> None:
    req = ChatCompletionsRequest(
        messages=[ChatMessage(role="user", content="hello")],
        tools=[
            ToolSpec(
                type="function",
                function=FunctionSpec(name="echo_tool", future_function_option=True),
            )
        ],
        reasoning_effort="high",
    )

    with pytest.raises(ValueError, match="future_function_option"):
        service().run_nonstream(req)


def test_refusal_is_preserved_when_all_assistant_tool_calls_are_bridged() -> None:
    adapter = CapturingAdapter()
    req = ChatCompletionsRequest(
        messages=[
            ChatMessage(role="user", content="weather in Kyoto"),
            ChatMessage(
                role="assistant",
                content=None,
                refusal="I cannot answer directly.",
                tool_calls=[
                    {
                        "id": "call_search",
                        "type": "function",
                        "function": {"name": "search_web", "arguments": '{"query":"Kyoto weather"}'},
                    }
                ],
            ),
            ChatMessage(role="tool", tool_call_id="call_search", content="legacy placeholder"),
        ],
        tools=[ToolSpec(type="function", function=FunctionSpec(name="search_web"))],
    )

    service(adapter).run_nonstream(req)

    assert adapter.calls[0]["input_payload"] == [
        {"role": "user", "content": "weather in Kyoto"},
        {
            "role": "assistant",
            "content": [{"type": "input_text", "text": "I cannot answer directly."}],
        },
        {"role": "user", "content": "Use web search to answer this query: Kyoto weather"},
    ]
