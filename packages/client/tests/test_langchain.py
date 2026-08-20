from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

import httpx
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    message_chunk_to_message,
    message_to_dict,
    messages_from_dict,
)
from langchain_openai import ChatOpenAI

from openai_compat_gateway_client import ChatOpenAICompat

STATE = {
    "mode": "responses",
    "response_items": [
        {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "opaque-encrypted-reasoning",
        }
    ],
}


def _completion(message: dict[str, Any], *, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.6-terra",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _llm(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> ChatOpenAICompat:
    transport = httpx.MockTransport(handler)
    return ChatOpenAICompat(
        model="gpt-5.6-terra",
        api_key="gateway-test-key",
        base_url="http://gateway.invalid/v1",
        http_client=httpx.Client(transport=transport),
        http_async_client=httpx.AsyncClient(transport=transport),
        **kwargs,
    )


def test_supported_private_api_contract_is_unchanged() -> None:
    expected = {
        "_use_responses_api": ["self", "payload"],
        "_get_request_payload": ["self", "input_", "stop", "kwargs"],
        "_create_chat_result": ["self", "response", "generation_info"],
        "_convert_chunk_to_generation_chunk": [
            "self",
            "chunk",
            "default_chunk_class",
            "base_generation_info",
        ],
    }
    for method_name, parameters in expected.items():
        assert list(inspect.signature(getattr(ChatOpenAI, method_name)).parameters) == parameters


def test_invoke_never_bypasses_gateway_chat_completions_endpoint() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json=_completion({"role": "assistant", "content": "result"}))

    # ChatOpenAI normally auto-selects the Responses API for this output mode.
    # ChatOpenAICompat must still target the gateway's compatibility endpoint;
    # the gateway owns the upstream Chat-vs-Responses routing itself.
    _llm(handler, output_version="responses/v1").invoke([HumanMessage(content="hello")])

    assert paths == ["/v1/chat/completions"]


def test_nonstream_state_is_saved_and_resent() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        message = (
            {"role": "assistant", "content": "first", "x_openai": STATE}
            if len(requests) == 1
            else {"role": "assistant", "content": "continued"}
        )
        return httpx.Response(200, json=_completion(message))

    llm = _llm(handler, reasoning_effort="medium")
    first = llm.invoke([HumanMessage(content="start")])
    second = llm.invoke([HumanMessage(content="start"), first, HumanMessage(content="continue")])

    assert first.additional_kwargs["x_openai"] == STATE
    assert requests[1]["messages"][1]["x_openai"] == STATE
    assert second.content == "continued"


def test_nonstream_merges_observability_fields_with_message_replay_state() -> None:
    top_level_state = {
        "citations": [{"url": "https://example.test/source", "title": "Source"}],
        "builtin_tool_events": [{"type": "web_search", "status": "completed"}],
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        response = _completion({"role": "assistant", "content": "first", "x_openai": STATE})
        response["x_openai"] = top_level_state
        return httpx.Response(200, json=response)

    assistant = _llm(handler).invoke([HumanMessage(content="start")])

    assert assistant.additional_kwargs["x_openai"] == {**top_level_state, **STATE}


def test_bind_tools_two_turn_round_trip_and_serialization() -> None:
    requests: list[dict[str, Any]] = []
    state_with_call = {
        **STATE,
        "response_items": [
            *STATE["response_items"],
            {
                "type": "function_call",
                "call_id": "call_weather",
                "name": "get_weather",
                "arguments": '{"city":"Kyoto"}',
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                200,
                json=_completion(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Kyoto"}',
                                },
                            }
                        ],
                        "x_openai": state_with_call,
                    },
                    finish_reason="tool_calls",
                ),
            )
        return httpx.Response(
            200,
            json=_completion({"role": "assistant", "content": "Kyoto is sunny."}),
        )

    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    bound = _llm(handler, reasoning_effort="medium").bind_tools([tool])
    user = HumanMessage(content="Check Kyoto weather with the tool.")
    assistant = bound.invoke([user])

    serialized = message_to_dict(assistant)
    restored = messages_from_dict([serialized])[0]
    assert isinstance(restored, AIMessage)
    assert restored.additional_kwargs["x_openai"] == state_with_call
    assert restored.tool_calls == assistant.tool_calls

    tool_result = ToolMessage(content='{"weather":"sunny"}', tool_call_id=assistant.tool_calls[0]["id"])
    final = bound.invoke([user, restored, tool_result])

    assistant_payload = requests[1]["messages"][1]
    assert assistant_payload["x_openai"] == state_with_call
    assert assistant_payload["tool_calls"][0]["id"] == "call_weather"
    assert requests[1]["messages"][2]["role"] == "tool"
    assert requests[1]["messages"][2]["tool_call_id"] == "call_weather"
    assert final.content == "Kyoto is sunny."


def test_builtin_tool_extra_body_is_sent() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_completion({"role": "assistant", "content": "result"}))

    llm = _llm(handler, extra_body={"x_builtin_tools": {"web_search": True}})
    llm.invoke([HumanMessage(content="Search the web")])

    assert requests[0]["x_builtin_tools"] == {"web_search": True}


def test_stream_state_is_saved_and_resent() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if body.get("stream"):
            chunks = [
                {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-5.6-terra",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "hello"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-5.6-terra",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"x_openai": STATE},
                            "finish_reason": "stop",
                        }
                    ],
                    "x_openai": STATE,
                },
            ]
            content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=_completion({"role": "assistant", "content": "continued"}))

    llm = _llm(handler, reasoning_effort="medium")
    stream = iter(llm.stream([HumanMessage(content="start")]))
    combined = next(stream)
    for chunk in stream:
        combined += chunk
    assistant = message_chunk_to_message(combined)

    assert assistant.additional_kwargs["x_openai"] == STATE
    llm.invoke([HumanMessage(content="start"), assistant, HumanMessage(content="continue")])
    assert requests[1]["messages"][1]["x_openai"] == STATE


def test_stream_merges_observability_fields_with_final_delta_replay_state() -> None:
    observability = {
        "code_interpreter_outputs": [{"type": "logs", "data": {"logs": "redacted"}}],
        "builtin_tool_events": [{"type": "code_interpreter", "status": "completed"}],
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        chunks = [
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-5.6-terra",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "done"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-5.6-terra",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"x_openai": STATE},
                        "finish_reason": "stop",
                    }
                ],
                "x_openai": {**observability, **STATE},
            },
        ]
        content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})

    stream = iter(_llm(handler).stream([HumanMessage(content="start")]))
    combined = next(stream)
    for chunk in stream:
        combined += chunk
    assistant = message_chunk_to_message(combined)

    assert assistant.additional_kwargs["x_openai"] == {**observability, **STATE}
