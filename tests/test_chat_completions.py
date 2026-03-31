from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api.chat_completions import get_service
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


class ExecutingToolExecutor:
    def __init__(self) -> None:
        self.calls = []

    def has(self, name: str) -> bool:
        return name == "lookup_profile"

    def execute(self, name: str, arguments_json: str) -> str:
        self.calls.append((name, arguments_json))
        return '{"role":"admin"}'


class NullToolExecutor:
    def has(self, _name: str) -> bool:
        return False


def _override_service(service: ChatService) -> TestClient:
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def test_function_call_roundtrip() -> None:
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
            ),
            SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="output_text", text="ユーザーは管理者です", annotations=[])],
                    )
                ],
                usage={"total_tokens": 12},
            ),
        ]
    )
    tool_executor = ExecutingToolExecutor()
    service = ChatService(
        adapter=adapter,
        tool_executor=tool_executor,
        audit_logger=DummyAuditLogger(),
        default_model="gpt-5.4-mini",
    )
    client = _override_service(service)

    try:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.4-mini",
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
                                "required": ["user_id"],
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
    assert tool_executor.calls == [("lookup_profile", '{"user_id":"u1"}')]
    assert adapter.calls[1]["input_payload"] == [
        {"role": "user", "content": "u1を見て"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup_profile",
            "arguments": '{"user_id":"u1"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": '{"role":"admin"}'},
    ]
    assert data["choices"][0]["message"]["content"] == "ユーザーは管理者です"
    assert data["usage"] == {"total_tokens": 12}


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
