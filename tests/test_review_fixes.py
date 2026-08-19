from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.mappers.request_mapper import to_responses_input
from app.schemas.compat import BuiltinToolsConfig, ChatCompletionsRequest, ChatMessage, FunctionSpec, ToolSpec
from app.services.chat_service import ChatService


class CapturingAdapter:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create_response(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return []
        return SimpleNamespace(output=[], usage=None, id="resp_test", status="completed", incomplete_details=None)


class NullExecutor:
    pass


class NullAuditLogger:
    def log_chat(self, req, normalized):
        return None


def make_service(adapter: CapturingAdapter | None = None) -> ChatService:
    return ChatService(
        adapter=adapter or CapturingAdapter(),
        tool_executor=NullExecutor(),
        audit_logger=NullAuditLogger(),
        default_model="gpt-5.6",
    )


def test_stream_options_split_usage_and_obfuscation_for_responses() -> None:
    adapter = CapturingAdapter()
    service = make_service(adapter)
    req = ChatCompletionsRequest(
        messages=[ChatMessage(role="user", content="hello")],
        stream=True,
        stream_options={"include_usage": True, "include_obfuscation": False},
        x_builtin_tools=BuiltinToolsConfig(web_search=True),
    )

    list(service.run_stream(req))

    assert adapter.calls[0]["stream_options"] == {"include_obfuscation": False}


def test_unknown_stream_option_is_rejected_in_responses_mode() -> None:
    service = make_service()
    req = ChatCompletionsRequest(
        messages=[ChatMessage(role="user", content="hello")],
        stream=True,
        stream_options={"future_option": True},
        x_builtin_tools=BuiltinToolsConfig(web_search=True),
    )

    with pytest.raises(ValueError, match="unsupported stream_options fields: future_option"):
        service.run_stream(req)


def test_stream_options_are_rejected_when_stream_is_false() -> None:
    service = make_service()
    req = ChatCompletionsRequest(
        messages=[ChatMessage(role="user", content="hello")],
        stream_options={"include_obfuscation": False},
        x_builtin_tools=BuiltinToolsConfig(web_search=True),
    )

    with pytest.raises(ValueError, match="stream_options with stream=false"):
        service.run_nonstream(req)


def test_unknown_tool_wrapper_field_survives_pydantic_and_is_rejected() -> None:
    service = make_service()
    tool = ToolSpec(
        type="function",
        function=FunctionSpec(name="echo_tool"),
        future_wrapper_field="value",
    )
    assert tool.model_extra == {"future_wrapper_field": "value"}

    req = ChatCompletionsRequest(
        messages=[ChatMessage(role="user", content="hello")],
        tools=[tool],
        reasoning_effort="high",
    )

    with pytest.raises(ValueError, match="unsupported tool wrapper fields: future_wrapper_field"):
        service.run_nonstream(req)


def test_unknown_tool_call_wrapper_field_is_rejected() -> None:
    service = make_service()
    req = ChatCompletionsRequest(
        messages=[
            ChatMessage(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo_tool", "arguments": "{}"},
                        "future_field": "value",
                    }
                ],
            )
        ],
    )

    with pytest.raises(ValueError, match="unsupported assistant tool_call fields: future_field"):
        service.run_nonstream(req)


def test_unknown_tool_call_function_field_is_rejected() -> None:
    service = make_service()
    req = ChatCompletionsRequest(
        messages=[
            ChatMessage(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "echo_tool",
                            "arguments": "{}",
                            "future_field": "value",
                        },
                    }
                ],
            )
        ],
    )

    with pytest.raises(ValueError, match="unsupported assistant tool_call.function fields: future_field"):
        service.run_nonstream(req)


@pytest.mark.parametrize(
    ("part", "message"),
    [
        (
            {"type": "text", "text": "hello", "future_field": "value"},
            "Chat Completions text content part contains unsupported fields: future_field",
        ),
        (
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png", "future_field": "value"},
            },
            "Chat Completions image_url object contains unsupported fields: future_field",
        ),
        (
            {
                "type": "file",
                "file": {"file_id": "file_123", "future_field": "value"},
            },
            "Chat Completions file object contains unsupported fields: future_field",
        ),
        (
            {
                "type": "text",
                "text": "hello",
                "prompt_cache_breakpoint": {"mode": "explicit", "future_field": "value"},
            },
            "Chat Completions prompt_cache_breakpoint contains unsupported fields: future_field",
        ),
    ],
)
def test_nested_unknown_content_fields_are_rejected(part: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        to_responses_input([{"role": "user", "content": [part]}])


def test_missing_text_value_is_rejected_instead_of_normalized_to_empty_string() -> None:
    with pytest.raises(ValueError, match="text content part requires string field 'text'"):
        to_responses_input([{"role": "user", "content": [{"type": "text"}]}])


def test_missing_refusal_value_is_rejected_instead_of_normalized_to_empty_string() -> None:
    with pytest.raises(ValueError, match="refusal content part requires string field 'refusal'"):
        to_responses_input([{"role": "assistant", "content": [{"type": "refusal"}]}])


def test_single_dict_content_keeps_top_level_assistant_refusal() -> None:
    items = to_responses_input(
        [
            {
                "role": "assistant",
                "content": {"type": "text", "text": "visible text"},
                "refusal": "refusal text",
            }
        ]
    )

    assert items == [
        {
            "role": "assistant",
            "content": [
                {"type": "input_text", "text": "visible text"},
                {"type": "input_text", "text": "refusal text"},
            ],
        }
    ]
