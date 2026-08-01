from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
import uvicorn
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    message_chunk_to_message,
    message_to_dict,
    messages_from_dict,
)
from openai_compat_gateway_client import ChatOpenAICompat

from app.adapters.openai_responses import OpenAIResponsesAdapter
from app.api.chat_completions import get_service
from app.config import settings
from app.main import app
from app.services.chat_service import ChatService

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.getenv("RUN_OPENAI_E2E") or not settings.openai_api_key,
        reason="requires RUN_OPENAI_E2E=1 and OPENAI_API_KEY",
    ),
]


class DummyAuditLogger:
    def log_chat(self, _req, _normalized) -> dict:
        return {}


class NullToolExecutor:
    def has(self, _name: str) -> bool:
        return False


def _item_kind(item: dict[str, Any]) -> str:
    return str(item.get("type") or f"message:{item.get('role', 'unknown')}")


def _safe_item(item: dict[str, Any]) -> dict[str, str | None]:
    """Retain only structure and a digest, never prompt/tool/reasoning contents."""

    encoded = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "kind": _item_kind(item),
        "id": item.get("id"),
        "call_id": item.get("call_id"),
        "digest": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def _safe_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


class CapturingOpenAIResponsesAdapter(OpenAIResponsesAdapter):
    """Capture safe upstream request fingerprints for transcript-order assertions."""

    def __init__(self, api_key: str | None = None):
        super().__init__(api_key=api_key)
        self.calls: list[dict[str, Any]] = []

    def create_response(self, **kwargs):
        self.calls.append(
            {
                "input_items": [_safe_item(item) for item in kwargs["input_payload"]],
                "tool_types": [tool.get("type") for tool in (kwargs.get("tools") or [])],
                "stream": bool(kwargs.get("stream")),
            }
        )
        return super().create_response(**kwargs)


@contextmanager
def _gateway(adapter: OpenAIResponsesAdapter) -> Iterator[str]:
    service = ChatService(
        adapter=adapter,
        tool_executor=NullToolExecutor(),
        audit_logger=DummyAuditLogger(),
        default_model=settings.openai_model_default,
    )
    app.dependency_overrides[get_service] = lambda: service

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()

    try:
        for _ in range(500):
            if server.started:
                break
            if not thread.is_alive():
                raise RuntimeError("local E2E gateway stopped before startup")
            time.sleep(0.01)
        else:
            raise RuntimeError("local E2E gateway did not start")
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        app.dependency_overrides.clear()


def _llm(base_url: str, **kwargs: Any) -> ChatOpenAICompat:
    return ChatOpenAICompat(
        model=settings.openai_model_default,
        base_url=base_url,
        api_key="local-gateway-e2e-key",
        reasoning_effort="medium",
        http_client=httpx.Client(trust_env=False),
        http_async_client=httpx.AsyncClient(trust_env=False),
        **kwargs,
    )


def _round_trip(message: AIMessage) -> AIMessage:
    stored = json.dumps(message_to_dict(message), ensure_ascii=False)
    restored = messages_from_dict([json.loads(stored)])[0]
    assert isinstance(restored, AIMessage)
    return restored


def _response_items(message: AIMessage) -> list[dict[str, Any]]:
    state = message.additional_kwargs.get("x_openai") or {}
    items = state.get("response_items") or []
    if not items:
        pytest.fail("x_openai.response_items is empty")
    return items


def _assert_item_type(items: list[dict[str, Any]], required: str) -> None:
    kinds = [_item_kind(item) for item in items]
    if required not in kinds:
        pytest.fail(f"missing response item type {required!r}; observed types: {kinds}")


def _assert_replayed_at(
    captured_call: dict[str, Any],
    expected_items: list[dict[str, Any]],
    start: int,
) -> None:
    expected = [_safe_item(item) for item in expected_items]
    observed = captured_call["input_items"][start : start + len(expected)]
    if observed != expected:
        expected_kinds = [item["kind"] for item in expected]
        observed_kinds = [item["kind"] for item in observed]
        pytest.fail(
            f"Responses items were not replayed at input index {start}; "
            f"expected types {expected_kinds}, observed {observed_kinds}"
        )


def _iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_strings(nested)


def _urls(value: Any) -> list[str]:
    found: list[str] = []
    for text in _iter_strings(value):
        found.extend(re.findall(r"https?://[^\s<>\"]+", text))
    return found


def test_real_openai_langchain_reasoning_custom_tool_round_trip() -> None:
    adapter = OpenAIResponsesAdapter(api_key=settings.openai_api_key)
    with _gateway(adapter) as base_url:
        llm = _llm(base_url)
        tool = {
            "type": "function",
            "function": {
                "name": "lookup_order_status",
                "description": "Return the shipment status for an order id.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }
        bound = llm.bind_tools([tool], tool_choice="lookup_order_status")
        user = HumanMessage(
            content="Call lookup_order_status exactly once for ord_314, then report the tracking code and status."
        )
        assistant = bound.invoke([user])

        assert assistant.tool_calls
        assert assistant.additional_kwargs["x_openai"]["response_items"]
        call = assistant.tool_calls[0]
        assert call["name"] == "lookup_order_status"
        assert call["args"]["order_id"] == "ord_314"

        tool_result = ToolMessage(
            content=json.dumps(
                {
                    "order_id": "ord_314",
                    "status": "shipped",
                    "tracking_code": "ZX-31415",
                }
            ),
            tool_call_id=call["id"],
        )
        final = llm.bind_tools([tool], tool_choice="auto").invoke([user, assistant, tool_result])

        assert "ZX-31415" in str(final.content)
        assert "shipped" in str(final.content).lower()


def test_real_web_search_replay_and_custom_tool_mixed_transcript() -> None:
    adapter = CapturingOpenAIResponsesAdapter(api_key=settings.openai_api_key)
    with _gateway(adapter) as base_url:
        llm = _llm(
            base_url,
            extra_body={"x_builtin_tools": {"web_search": True}},
        )
        first_user = HumanMessage(
            content=(
                "Use web search to find one current Python release page on python.org. "
                "Reply with exactly two lines: RESULT_URL:<absolute URL> and RESULT_TITLE:<page title>."
            )
        )
        first = llm.invoke([first_user], tool_choice="required")
        first_items = _response_items(first)
        _assert_item_type(first_items, "web_search_call")
        assert adapter.calls[0]["stream"] is False
        assert "web_search" in adapter.calls[0]["tool_types"]

        first_state = first.additional_kwargs["x_openai"]
        builtin_events = first_state.get("builtin_tool_events") or []
        if not any(event.get("type") == "web_search" for event in builtin_events):
            pytest.fail("x_openai.builtin_tool_events is missing the real web_search event")

        match = re.search(r"RESULT_URL:\s*(https?://[^\s<>]+)", str(first.content))
        if not match:
            pytest.fail("web search answer did not contain the requested RESULT_URL marker")
        selected_url = match.group(1).rstrip(".,;:)]}")
        selected_host = urlparse(selected_url).hostname
        if not selected_host:
            pytest.fail("the selected web search URL did not contain a hostname")

        restored_first = _round_trip(first)
        if _safe_digest(restored_first.additional_kwargs["x_openai"]) != _safe_digest(first_state):
            pytest.fail("web search x_openai state changed during JSON persistence")
        returned_urls = _urls(first_state)
        if returned_urls and _urls(restored_first.additional_kwargs["x_openai"]) != returned_urls:
            pytest.fail("web search citations or sources changed during JSON persistence")

        capture_tool = {
            "type": "function",
            "function": {
                "name": "capture_selected_host",
                "description": "Persist the hostname selected in the previous web-search turn.",
                "parameters": {
                    "type": "object",
                    "properties": {"hostname": {"type": "string"}},
                    "required": ["hostname"],
                },
            },
        }
        second_user = HumanMessage(
            content=(
                "Do not search again. Call capture_selected_host exactly once with only the hostname "
                "from the RESULT_URL you selected in the previous turn."
            )
        )
        bound = llm.bind_tools([capture_tool], tool_choice="capture_selected_host")
        custom_assistant = bound.invoke([first_user, restored_first, second_user])
        if not custom_assistant.tool_calls:
            pytest.fail("the mixed built-in/custom turn did not emit a LangChain tool call")
        call = custom_assistant.tool_calls[0]
        assert call["name"] == "capture_selected_host"
        assert str(call["args"]["hostname"]).lower() == selected_host.lower()
        _assert_replayed_at(adapter.calls[1], first_items, start=1)

        custom_items = _response_items(custom_assistant)
        _assert_item_type(custom_items, "function_call")
        if any(_item_kind(item) == "web_search_call" for item in custom_items):
            pytest.fail("the forced custom-tool turn unexpectedly ran web search again")
        restored_custom = _round_trip(custom_assistant)

        receipt = f"receipt-{secrets.token_hex(8)}"
        tool_message = ToolMessage(
            content=json.dumps({"hostname": selected_host, "receipt": receipt}),
            tool_call_id=call["id"],
        )
        third_user = HumanMessage(
            content="Without calling another tool, reply with the saved hostname and receipt from the tool result."
        )
        final = llm.bind_tools([capture_tool], tool_choice="none").invoke(
            [
                first_user,
                restored_first,
                second_user,
                restored_custom,
                tool_message,
                third_user,
            ]
        )

        _assert_replayed_at(adapter.calls[2], first_items, start=1)
        custom_start = len(first_items) + 2
        _assert_replayed_at(adapter.calls[2], custom_items, start=custom_start)
        tool_output_index = custom_start + len(custom_items)
        expected_tool_output = {
            "type": "function_call_output",
            "call_id": call["id"],
            "output": tool_message.content,
        }
        if adapter.calls[2]["input_items"][tool_output_index] != _safe_item(expected_tool_output):
            pytest.fail("custom ToolMessage output was not preserved after built-in state replay")
        if selected_host.lower() not in str(final.content).lower() or receipt not in str(final.content):
            pytest.fail("final answer did not use both the prior web result and custom ToolMessage result")


def test_real_code_interpreter_streaming_hidden_value_replay() -> None:
    adapter = CapturingOpenAIResponsesAdapter(api_key=settings.openai_api_key)
    with _gateway(adapter) as base_url:
        llm = _llm(
            base_url,
            extra_body={"x_builtin_tools": {"code_interpreter": True}},
        )
        first_user = HumanMessage(
            content=(
                "Use Code Interpreter and Python secrets.token_hex(16) to generate a random value. "
                "Print exactly STATE_NONCE:<value> in the Python output, but do not reveal the value "
                "in your assistant text; the assistant text must be only STATE_READY."
            )
        )
        chunks = iter(llm.stream([first_user], tool_choice="required"))
        combined = next(chunks)
        for chunk in chunks:
            combined += chunk
        first = message_chunk_to_message(combined)

        first_items = _response_items(first)
        _assert_item_type(first_items, "code_interpreter_call")
        assert adapter.calls[0]["stream"] is True
        assert "code_interpreter" in adapter.calls[0]["tool_types"]
        if "STATE_READY" not in str(first.content):
            pytest.fail("streaming Code Interpreter turn did not return the completion marker")

        state_text = "\n".join(_iter_strings(first.additional_kwargs["x_openai"]))
        nonce_match = re.search(r"STATE_NONCE:([0-9a-f]{32})", state_text)
        if not nonce_match:
            kinds = [_item_kind(item) for item in first_items]
            pytest.fail(f"Code Interpreter output is missing STATE_NONCE; observed item types: {kinds}")
        nonce = nonce_match.group(1)
        if nonce in str(first.content):
            pytest.fail("the hidden Code Interpreter nonce leaked into assistant text")

        code_outputs = first.additional_kwargs["x_openai"].get("code_interpreter_outputs") or []
        if not code_outputs:
            pytest.fail("x_openai.code_interpreter_outputs is empty after streaming aggregation")
        encrypted_reasoning = [
            item for item in first_items if _item_kind(item) == "reasoning" and item.get("encrypted_content")
        ]

        restored_first = _round_trip(first)
        restored_items = _response_items(restored_first)
        assert [_safe_item(item) for item in restored_items] == [_safe_item(item) for item in first_items]
        if encrypted_reasoning:
            restored_encrypted = [
                item for item in restored_items if _item_kind(item) == "reasoning" and item.get("encrypted_content")
            ]
            assert [_safe_item(item) for item in restored_encrypted] == [
                _safe_item(item) for item in encrypted_reasoning
            ]

        followup = HumanMessage(
            content=(
                "Use Code Interpreter to calculate SHA-256 of the random value generated in the previous turn. "
                "Reply exactly STATE_SHA256:<lowercase hex digest>."
            )
        )
        second = llm.invoke([first_user, restored_first, followup], tool_choice="required")
        _assert_replayed_at(adapter.calls[1], first_items, start=1)
        second_items = _response_items(second)
        _assert_item_type(second_items, "code_interpreter_call")

        expected_hash = hashlib.sha256(nonce.encode()).hexdigest()
        hash_match = re.search(r"STATE_SHA256:([0-9a-f]{64})", str(second.content))
        if not hash_match:
            pytest.fail("second Code Interpreter turn did not return the requested hash marker")
        if hash_match.group(1) != expected_hash:
            pytest.fail("second Code Interpreter turn did not hash the hidden first-turn value")
