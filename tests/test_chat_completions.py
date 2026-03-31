from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.chat_completions import get_service
from app.adapters.openai_responses import OpenAIResponsesAdapter
from app.config import settings
from app.main import app
from app.services.chat_service import ChatService
class DummyAuditLogger:
    def __init__(self) -> None:
        self.logged = []

    def log_chat(self, req, normalized) -> dict:
        self.logged.append((req, normalized))
        return {}


class CapturingAdapter:
    def __init__(self, responses) -> None:
        self.calls = []
        self.responses = list(responses)

    def create_response(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class NullToolExecutor:
    def has(self, _name: str) -> bool:
        return False


def _override_service(service: ChatService) -> TestClient:
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def test_unknown_custom_tool_is_returned_to_client_for_execution() -> None:
    adapter = CapturingAdapter(
        responses=[
            SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        call_id="call_1",
                        name="lookup_profile",
                        arguments='{"user_id":"u1"}',
                    )
                ],
                usage={"total_tokens": 3},
            )
        ]
    )
    service = ChatService(
        adapter=adapter,
        tool_executor=NullToolExecutor(),
        audit_logger=DummyAuditLogger(),
        default_model="gpt-5.4-mini",
    )
    client = _override_service(service)

    try:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "u1を見て"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_profile",
                            "description": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {"user_id": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        )
    finally:
        app.dependency_overrides.clear()

    data = resp.json()
    assert resp.status_code == 200
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert data["choices"][0]["message"]["content"] == ""
    assert data["choices"][0]["message"]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "lookup_profile",
                "arguments": '{"user_id":"u1"}',
            },
        }
    ]


def test_followup_tool_messages_are_replayed_to_responses_api() -> None:
    adapter = CapturingAdapter(
        responses=[
            SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="output_text", text="ユーザーは管理者です", annotations=[])],
                    )
                ],
                usage={"total_tokens": 12},
            )
        ]
    )
    service = ChatService(
        adapter=adapter,
        tool_executor=NullToolExecutor(),
        audit_logger=DummyAuditLogger(),
        default_model="gpt-5.4-mini",
    )
    client = _override_service(service)

    try:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "u1を見て"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup_profile",
                                    "arguments": '{"user_id":"u1"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": '{"role":"admin"}',
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_profile",
                            "description": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {"user_id": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        )
    finally:
        app.dependency_overrides.clear()

    data = resp.json()
    assert resp.status_code == 200
    assert adapter.calls[0]["input_payload"] == [
        {"role": "user", "content": "u1を見て"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup_profile",
            "arguments": '{"user_id":"u1"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": '{"role":"admin"}'},
    ]
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["choices"][0]["message"]["content"] == "ユーザーは管理者です"


def test_web_search_bridge_surfaces_citations_and_legacy_steps() -> None:
    adapter = CapturingAdapter(
        responses=[
            SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="web_search_call",
                        id="ws_1",
                        status="completed",
                        action=SimpleNamespace(
                            query="tokyo weather",
                            sources=[
                                SimpleNamespace(
                                    type="url",
                                    url="https://example.com/weather",
                                    title="Weather Source",
                                )
                            ],
                        ),
                    ),
                    SimpleNamespace(
                        type="message",
                        content=[
                            SimpleNamespace(
                                type="output_text",
                                text="晴れです",
                                annotations=[
                                    SimpleNamespace(
                                        type="url_citation",
                                        url="https://example.com/detail",
                                        title="Forecast Detail",
                                    )
                                ],
                            )
                        ],
                    ),
                ],
                usage={"total_tokens": 8},
            )
        ]
    )
    service = ChatService(
        adapter=adapter,
        tool_executor=NullToolExecutor(),
        audit_logger=DummyAuditLogger(),
        default_model="gpt-5.4-mini",
    )
    client = _override_service(service)

    try:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "東京の天気は?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "description": "Search the web",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        )
    finally:
        app.dependency_overrides.clear()

    data = resp.json()
    assert resp.status_code == 200
    assert adapter.calls[0]["tools"] == [{"type": "web_search"}]
    assert adapter.calls[0]["include"] == ["web_search_call.action.sources"]
    assert data["choices"][0]["message"]["content"] == "晴れです"
    assert data["x_openai"]["citations"] == [
        {"url": "https://example.com/weather", "title": "Weather Source"},
        {"url": "https://example.com/detail", "title": "Forecast Detail"},
    ]
    assert data["x_openai"]["legacy_steps"][0]["tool_calls"][0]["function"]["name"] == "search_web"
    assert data["x_openai"]["legacy_steps"][1]["name"] == "search_web"


def test_stream_returns_chunks_and_done() -> None:
    adapter = CapturingAdapter(
        responses=[
            [
                SimpleNamespace(type="response.output_text.delta", delta="こんにちは"),
                SimpleNamespace(
                    type="response.output_item.done",
                    item=SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="output_text", text="こんにちは", annotations=[])],
                    ),
                ),
                SimpleNamespace(type="response.completed"),
            ]
        ]
    )
    service = ChatService(
        adapter=adapter,
        tool_executor=NullToolExecutor(),
        audit_logger=DummyAuditLogger(),
        default_model="gpt-5.4-mini",
    )
    client = _override_service(service)

    try:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "挨拶して"}],
                "stream": True,
            },
        ) as resp:
            body = "".join(resp.iter_text())
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = body.replace("\\n", "\n")
    events = [line[len("data: ") :] for line in body.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    first_payload = json.loads(events[0])
    final_payload = json.loads(events[-2])
    assert first_payload["object"] == "chat.completion.chunk"
    assert final_payload["choices"][0]["finish_reason"] == "stop"
    assert final_payload["x_openai"]["assistant_text"] == "こんにちは"


def test_stream_tool_call_finishes_with_tool_calls() -> None:
    adapter = CapturingAdapter(
        responses=[
            [
                SimpleNamespace(
                    type="response.function_call_arguments.delta",
                    call_id="call_1",
                    name="lookup_profile",
                    delta='{"user_id":"u1"}',
                ),
                SimpleNamespace(
                    type="response.output_item.done",
                    item=SimpleNamespace(
                        type="function_call",
                        call_id="call_1",
                        name="lookup_profile",
                        arguments='{"user_id":"u1"}',
                    ),
                ),
                SimpleNamespace(type="response.completed"),
            ]
        ]
    )
    service = ChatService(
        adapter=adapter,
        tool_executor=NullToolExecutor(),
        audit_logger=DummyAuditLogger(),
        default_model="gpt-5.4-mini",
    )
    client = _override_service(service)

    try:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "u1を見て"}],
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_profile",
                            "description": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {"user_id": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        ) as resp:
            body = "".join(resp.iter_text())
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    events = [line[len("data: ") :] for line in body.replace("\\n", "\n").splitlines() if line.startswith("data: ")]
    final_payload = json.loads(events[-2])
    assert final_payload["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.e2e
@pytest.mark.skipif(
    not os.getenv("RUN_OPENAI_E2E") or not settings.openai_api_key,
    reason="requires RUN_OPENAI_E2E=1 and OPENAI_API_KEY",
)
def test_real_openai_api_custom_tool_roundtrip() -> None:
    observed_calls: list[tuple[str, str]] = []

    def lookup_order_status(order_id: str) -> dict[str, str]:
        observed_calls.append(("lookup_order_status", order_id))
        return {
            "order_id": order_id,
            "status": "shipped",
            "tracking_code": "ZX-31415",
        }

    service = ChatService(
        adapter=OpenAIResponsesAdapter(api_key=settings.openai_api_key),
        tool_executor=NullToolExecutor(),
        audit_logger=DummyAuditLogger(),
        default_model=settings.openai_model_default,
    )
    client = _override_service(service)

    try:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": settings.openai_model_default,
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "lookup_order_status"},
                },
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "lookup_order_status を必ず1回だけ使って order_id ord_314 を確認し、"
                            "最終回答は tracking_code と status を短く返してください。"
                        ),
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_order_status",
                            "description": "Return the shipment status for an order id.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "order_id": {
                                        "type": "string",
                                        "description": "Order identifier such as ord_314",
                                    }
                                },
                                "required": ["order_id"],
                            },
                        },
                    }
                ],
            },
        )
    finally:
        app.dependency_overrides.clear()

    data = resp.json()
    tool_calls = data["choices"][0]["message"]["tool_calls"]

    assert resp.status_code == 200, data
    assert observed_calls == []
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert tool_calls[0]["function"]["name"] == "lookup_order_status"
