from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.schemas.compat import BuiltinToolsConfig, ChatCompletionsRequest, ChatMessage
from app.services.chat_service import ChatService


class DummyAuditLogger:
    def __init__(self) -> None:
        self.logged = []

    def log_chat(self, req, normalized) -> dict:
        self.logged.append((req, normalized))
        return {}


class DummyToolExecutor:
    def has(self, _name: str) -> bool:
        return False


class CapturingAdapter:
    def __init__(self) -> None:
        self.calls = []

    def create_response(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output=[], usage={"total_tokens": 1})


class ChatServiceTests(unittest.TestCase):
    def _make_request(self, builtin_tools: BuiltinToolsConfig | None = None) -> ChatCompletionsRequest:
        return ChatCompletionsRequest(
            messages=[ChatMessage(role="user", content="hello")],
            x_builtin_tools=builtin_tools,
        )

    def test_nonstream_web_search_omits_results_include_by_default(self) -> None:
        adapter = CapturingAdapter()
        service = ChatService(
            adapter,
            DummyToolExecutor(),
            DummyAuditLogger(),
            default_model="gpt-5.4-mini",
        )

        service.run_nonstream(self._make_request(BuiltinToolsConfig(web_search=True)))

        self.assertEqual(
            adapter.calls[0]["include"],
            ["web_search_call.action.sources"],
        )

    def test_nonstream_web_search_results_include_can_be_enabled(self) -> None:
        adapter = CapturingAdapter()
        service = ChatService(
            adapter,
            DummyToolExecutor(),
            DummyAuditLogger(),
            default_model="gpt-5.4-mini",
            include_web_search_results=True,
        )

        service.run_nonstream(self._make_request(BuiltinToolsConfig(web_search=True)))

        self.assertEqual(
            adapter.calls[0]["include"],
            ["web_search_call.action.sources", "web_search_call.results"],
        )

    def test_stream_passes_safe_include_set(self) -> None:
        adapter = CapturingAdapter()
        service = ChatService(
            adapter,
            DummyToolExecutor(),
            DummyAuditLogger(),
            default_model="gpt-5.4-mini",
        )

        service.run_stream(
            self._make_request(
                BuiltinToolsConfig(
                    web_search=True,
                    file_search={"vector_store_ids": ["vs_123"]},
                    code_interpreter=True,
                )
            )
        )

        self.assertEqual(
            adapter.calls[0]["include"],
            [
                "web_search_call.action.sources",
                "file_search_call.results",
                "code_interpreter_call.outputs",
            ],
        )
        self.assertTrue(adapter.calls[0]["stream"])


if __name__ == "__main__":
    unittest.main()
