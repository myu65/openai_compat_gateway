from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.schemas.compat import BuiltinToolsConfig, ChatCompletionsRequest, ChatMessage, FunctionSpec, ToolSpec
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
    def __init__(self, responses=None) -> None:
        self.calls = []
        self.responses = list(responses or [SimpleNamespace(output=[], usage={"total_tokens": 1})])

    def create_response(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class ChatServiceTests(unittest.TestCase):
    def _make_request(self, builtin_tools: BuiltinToolsConfig | None = None) -> ChatCompletionsRequest:
        return ChatCompletionsRequest(
            messages=[ChatMessage(role="user", content="hello")],
            x_builtin_tools=builtin_tools,
        )

    def _make_service(self, adapter=None) -> ChatService:
        return ChatService(
            adapter or CapturingAdapter(),
            DummyToolExecutor(),
            DummyAuditLogger(),
            default_model="gpt-5.4-mini",
        )

    def test_no_tools_default_tool_choice_to_none(self) -> None:
        adapter = CapturingAdapter()
        service = self._make_service(adapter)

        req = self._make_request()
        service.run_nonstream(req)

        self.assertEqual(req.tool_choice, "none")
        self.assertEqual(adapter.calls[0]["tools"], [])
        self.assertEqual(adapter.calls[0]["tool_choice"], "none")

    def test_nonstream_web_search_omits_results_include_by_default(self) -> None:
        adapter = CapturingAdapter()
        service = self._make_service(adapter)

        service.run_nonstream(self._make_request(BuiltinToolsConfig(web_search=True)))

        self.assertEqual(
            adapter.calls[0]["include"],
            ["web_search_call.action.sources", "reasoning.encrypted_content"],
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
            [
                "web_search_call.action.sources",
                "web_search_call.results",
                "reasoning.encrypted_content",
            ],
        )

    def test_stream_passes_safe_include_set(self) -> None:
        adapter = CapturingAdapter()
        service = self._make_service(adapter)

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
                "reasoning.encrypted_content",
            ],
        )
        self.assertTrue(adapter.calls[0]["stream"])

    def test_bridge_tool_definitions_enable_builtin_web_search(self) -> None:
        adapter = CapturingAdapter()
        service = self._make_service(adapter)

        req = ChatCompletionsRequest(
            messages=[ChatMessage(role="user", content="search for tokyo weather")],
            tools=[
                ToolSpec(
                    type="function",
                    function=FunctionSpec(
                        name="search_web",
                        description="Search the web",
                        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                    ),
                )
            ],
        )

        normalized = service.run_nonstream(req)

        self.assertEqual(adapter.calls[0]["tools"], [{"type": "web_search"}])
        self.assertEqual(adapter.calls[0]["tool_choice"], "auto")
        self.assertEqual(normalized.bridge_executions[0].display_tool_name, "search_web")

    def test_explicit_none_is_preserved_when_tools_are_configured(self) -> None:
        req = ChatCompletionsRequest(
            messages=[ChatMessage(role="user", content="hello")],
            tool_choice="none",
            tools=[ToolSpec(type="function", function=FunctionSpec(name="echo_tool"))],
        )

        self.assertEqual(req.tool_choice, "none")

    def test_nonstream_client_tool_followup_replays_in_request_state(self) -> None:
        adapter = CapturingAdapter(
            responses=[
                SimpleNamespace(
                    output=[
                        SimpleNamespace(
                            type="message",
                            content=[SimpleNamespace(type="output_text", text="done", annotations=[])],
                        )
                    ],
                    usage={"total_tokens": 2},
                ),
            ]
        )
        service = self._make_service(adapter)

        req = ChatCompletionsRequest(
            messages=[
                ChatMessage(role="user", content="hello"),
                ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "echo_tool", "arguments": '{"text":"hello"}'},
                        }
                    ],
                ),
                ChatMessage(role="tool", content='{"echo":"ok"}', tool_call_id="call_123"),
            ],
            tool_choice="auto",
            tools=[
                ToolSpec(
                    type="function",
                    function=FunctionSpec(
                        name="echo_tool",
                        description="Echo text",
                        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
                    ),
                )
            ],
        )

        normalized = service.run_nonstream(req)

        self.assertEqual(normalized.assistant_text, "done")
        self.assertEqual(
            adapter.calls[0]["input_payload"],
            [
                {"role": "user", "content": "hello"},
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "echo_tool",
                    "arguments": '{"text":"hello"}',
                },
                {"type": "function_call_output", "call_id": "call_123", "output": '{"echo":"ok"}'},
            ],
        )
        self.assertEqual(adapter.calls[0]["tool_choice"], "auto")

    def test_chat_completions_function_tool_choice_is_normalized_for_responses_api(self) -> None:
        adapter = CapturingAdapter()
        service = self._make_service(adapter)

        req = ChatCompletionsRequest(
            messages=[ChatMessage(role="user", content="hello")],
            tool_choice={"type": "function", "function": {"name": "echo_tool"}},
            tools=[
                ToolSpec(
                    type="function",
                    function=FunctionSpec(
                        name="echo_tool",
                        description="Echo text",
                        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
                    ),
                )
            ],
        )

        service.run_nonstream(req)

        self.assertEqual(
            adapter.calls[0]["tool_choice"],
            {"type": "function", "name": "echo_tool"},
        )

    def test_metadata_is_preserved_in_responses_mode(self) -> None:
        adapter = CapturingAdapter()
        service = self._make_service(adapter)
        req = ChatCompletionsRequest(
            messages=[ChatMessage(role="user", content="hello")],
            metadata={"trace_id": "trace_123"},
        )

        service.run_nonstream(req)

        self.assertEqual(adapter.calls[0]["metadata"], {"trace_id": "trace_123"})

    def test_message_name_fails_instead_of_being_forwarded_or_dropped(self) -> None:
        service = self._make_service()
        req = ChatCompletionsRequest(
            messages=[ChatMessage(role="user", content="hello", name="alice")],
        )

        with self.assertRaisesRegex(ValueError, "message.name"):
            service.run_nonstream(req)

    def test_legacy_assistant_function_call_fails_instead_of_being_dropped(self) -> None:
        service = self._make_service()
        req = ChatCompletionsRequest(
            messages=[
                ChatMessage(
                    role="assistant",
                    content=None,
                    function_call={"name": "legacy_fn", "arguments": "{}"},
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "legacy assistant function_call"):
            service.run_nonstream(req)

    def test_assistant_audio_state_fails_instead_of_being_dropped(self) -> None:
        service = self._make_service()
        req = ChatCompletionsRequest(
            messages=[
                ChatMessage(
                    role="assistant",
                    content="hello",
                    audio={"id": "audio_123"},
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "assistant audio state"):
            service.run_nonstream(req)


if __name__ == "__main__":
    unittest.main()
