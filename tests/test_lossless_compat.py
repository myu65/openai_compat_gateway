from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
from langchain_core.messages import HumanMessage
from openai.types.chat import ChatCompletionChunk

from app.integrations.langchain import ChatOpenAICompat
from app.mappers.response_mapper import normalize_final_response
from app.mappers.stream_mapper import map_stream_events
from app.schemas.compat import (
    BuiltinToolsConfig,
    ChatCompletionsRequest,
    ChatMessage,
    FunctionSpec,
    OpenAIStateEnvelope,
    ToolSpec,
)
from app.services.chat_service import ChatService


class CapturingAdapter:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or SimpleNamespace(output=[], usage=None)

    def create_response(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class NativeAdapter:
    def __init__(self):
        self.calls = []

    def create_completion(self, payload, *, stream=False):
        self.calls.append((payload, stream))
        return {"choices": []}


class NullLogger:
    def log_chat(self, req, normalized):
        return None


class NullExecutor:
    pass


def service(adapter, native_adapter=None):
    return ChatService(
        adapter=adapter,
        native_adapter=native_adapter,
        tool_executor=NullExecutor(),
        audit_logger=NullLogger(),
        default_model="gpt-5.6",
    )


def test_reasoning_items_are_serialized_and_replayed_before_tool_output() -> None:
    state_items = [
        {
            "id": "rs_1",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "opaque-ciphertext",
        },
        {
            "id": "fc_1",
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"id":"u1"}',
            "status": "completed",
        },
    ]
    adapter = CapturingAdapter(
        SimpleNamespace(
            id="resp_2",
            status="completed",
            output=[SimpleNamespace(type="message", content=[])],
            usage=None,
            incomplete_details=None,
        )
    )
    req = ChatCompletionsRequest(
        messages=[
            ChatMessage(role="user", content="lookup u1"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"id":"u1"}'},
                    }
                ],
                x_openai=OpenAIStateEnvelope(response_items=state_items),
            ),
            ChatMessage(role="tool", tool_call_id="call_1", content='{"name":"Alice"}'),
        ],
        tools=[
            ToolSpec(
                type="function",
                function=FunctionSpec(
                    name="lookup",
                    parameters={"type": "object", "properties": {"id": {"type": "string"}}},
                ),
            )
        ],
        reasoning_effort="medium",
    )

    service(adapter).run_nonstream(req)

    assert adapter.calls[0]["input_payload"] == [
        {"role": "user", "content": "lookup u1"},
        *state_items,
        {"type": "function_call_output", "call_id": "call_1", "output": '{"name":"Alice"}'},
    ]
    assert adapter.calls[0]["reasoning"] == {"effort": "medium"}
    assert adapter.calls[0]["tools"][0]["strict"] is False
    assert "reasoning.encrypted_content" in adapter.calls[0]["include"]


def test_response_items_preserve_encrypted_reasoning() -> None:
    response = SimpleNamespace(
        id="resp_1",
        status="completed",
        incomplete_details=None,
        output=[
            SimpleNamespace(
                id="rs_1",
                type="reasoning",
                summary=[],
                encrypted_content="opaque-ciphertext",
            )
        ],
        usage=None,
    )

    normalized = normalize_final_response(response)

    assert normalized.response_id == "resp_1"
    assert normalized.response_items[0]["encrypted_content"] == "opaque-ciphertext"


def test_code_interpreter_always_has_a_container() -> None:
    adapter = CapturingAdapter()
    req = ChatCompletionsRequest(
        messages=[ChatMessage(role="user", content="calculate")],
        x_builtin_tools=BuiltinToolsConfig(code_interpreter=True),
    )

    service(adapter).run_nonstream(req)

    assert adapter.calls[0]["tools"] == [{"type": "code_interpreter", "container": {"type": "auto"}}]


def test_legacy_bridged_tool_transcript_does_not_leave_an_unmatched_function_call() -> None:
    adapter = CapturingAdapter()
    req = ChatCompletionsRequest(
        messages=[
            ChatMessage(role="user", content="weather in Kyoto"),
            ChatMessage(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call_search",
                        "type": "function",
                        "function": {"name": "search_web", "arguments": '{"query":"Kyoto weather"}'},
                    }
                ],
            ),
            # LangChain ToolMessage does not reliably preserve the tool name.
            ChatMessage(role="tool", tool_call_id="call_search", content="legacy placeholder"),
        ],
        tools=[ToolSpec(type="function", function=FunctionSpec(name="search_web"))],
    )

    service(adapter).run_nonstream(req)

    assert adapter.calls[0]["tools"] == [{"type": "web_search"}]
    assert not any(item.get("type") == "function_call" for item in adapter.calls[0]["input_payload"])
    assert adapter.calls[0]["input_payload"][-1] == {
        "role": "user",
        "content": "Use web search to answer this query: Kyoto weather",
    }


def test_stream_uses_real_sse_newlines_parallel_indices_and_state() -> None:
    response = SimpleNamespace(
        id="resp_1",
        status="completed",
        incomplete_details=None,
        usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        output=[
            SimpleNamespace(type="reasoning", encrypted_content="opaque", summary=[]),
            SimpleNamespace(type="function_call", call_id="call_a", name="a", arguments="{}"),
            SimpleNamespace(type="function_call", call_id="call_b", name="b", arguments="{}"),
        ],
    )
    events = [
        SimpleNamespace(type="response.function_call_arguments.delta", call_id="call_a", name="a", delta="{"),
        SimpleNamespace(type="response.function_call_arguments.delta", call_id="call_b", name="b", delta="{"),
        SimpleNamespace(type="response.completed", response=response),
    ]

    chunks = list(map_stream_events(events, "gpt-5.6", include_usage=True))
    payloads = [json.loads(chunk[len("data: ") :]) for chunk in chunks[:-1]]

    for payload in payloads:
        ChatCompletionChunk.model_validate(payload)

    assert chunks[-1] == "data: [DONE]\n\n"
    assert "\\n" not in chunks[0]
    assert payloads[1]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert payloads[2]["choices"][0]["delta"]["tool_calls"][0]["index"] == 1
    assert payloads[-2]["x_openai"]["response_items"][0]["encrypted_content"] == "opaque"
    assert payloads[-1]["choices"] == []
    assert payloads[-1]["usage"]["total_tokens"] == 3


def test_stream_maps_current_responses_function_delta_shape() -> None:
    function_item = SimpleNamespace(
        id="fc_item_1",
        type="function_call",
        call_id="call_1",
        name="lookup",
        arguments="",
    )
    response = SimpleNamespace(
        id="resp_1",
        status="completed",
        incomplete_details=None,
        usage=None,
        output=[
            SimpleNamespace(
                id="fc_item_1",
                type="function_call",
                call_id="call_1",
                name="lookup",
                arguments='{"id":"u1"}',
            )
        ],
    )
    events = [
        SimpleNamespace(type="response.output_item.added", output_index=0, item=function_item),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_item_1",
            output_index=0,
            delta='{"id":"u1"}',
        ),
        SimpleNamespace(type="response.completed", response=response),
    ]

    payloads = [
        json.loads(chunk[len("data: ") :])
        for chunk in map_stream_events(events, "gpt-5.6")
        if chunk.startswith("data: {")
    ]

    first_tool = payloads[1]["choices"][0]["delta"]["tool_calls"][0]
    args_delta = payloads[2]["choices"][0]["delta"]["tool_calls"][0]
    assert first_tool == {
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": ""},
    }
    assert args_delta == {"index": 0, "function": {"arguments": '{"id":"u1"}'}}


def test_auto_mode_splits_native_and_responses_paths() -> None:
    responses = CapturingAdapter()
    native = NativeAdapter()
    svc = service(responses, native)

    plain = ChatCompletionsRequest(messages=[ChatMessage(role="user", content="hello")])
    reasoned_tool = ChatCompletionsRequest(
        messages=[ChatMessage(role="user", content="hello")],
        reasoning_effort="medium",
        tools=[ToolSpec(type="function", function=FunctionSpec(name="lookup"))],
    )

    assert svc.select_mode(plain) == "chat_completions"
    assert svc.select_mode(reasoned_tool) == "responses"


def test_langchain_adapter_round_trips_gateway_state() -> None:
    llm = ChatOpenAICompat(
        model="gpt-5.6",
        api_key="test",
        base_url="http://gateway.invalid/v1",
        http_client=httpx.Client(trust_env=False),
        http_async_client=httpx.AsyncClient(trust_env=False),
    )
    state = {"response_items": [{"type": "reasoning", "encrypted_content": "opaque"}]}
    result = llm._create_chat_result(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "answer", "x_openai": state},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    message = result.generations[0].message

    payload = llm._get_request_payload([message, HumanMessage(content="next")])

    assert message.additional_kwargs["x_openai"] == state
    assert payload["messages"][0]["x_openai"] == state
