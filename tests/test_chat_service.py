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


class ExecutingToolExecutor:
    def has(self, name: str) -> bool:
        return name == "echo_tool"

    def execute(self, name: str, arguments_json: str) -> str:
        self.last_call = (name, arguments_json)
        return '{"echo":"ok"}'


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

    def test_bridge_tool_definitions_enable_builtin_web_search(self) -> None:
        adapter = CapturingAdapter()
        service = ChatService(
            adapter,
            DummyToolExecutor(),
            DummyAuditLogger(),
            default_model="gpt-5.4-mini",
        )

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
        self.assertEqual(normalized.bridge_executions[0].display_tool_name, "search_web")

    def test_nonstream_custom_function_loop_replays_in_request_state(self) -> None:
        adapter = CapturingAdapter(
            responses=[
                SimpleNamespace(
                    output=[
                        SimpleNamespace(
                            type="function_call",
                            call_id="call_123",
                            name="echo_tool",
                            arguments='{"text":"hello"}',
                        )
                    ],
                    usage={"total_tokens": 1},
                ),
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
        service = ChatService(
            adapter,
            ExecutingToolExecutor(),
            DummyAuditLogger(),
            default_model="gpt-5.4-mini",
        )

        req = ChatCompletionsRequest(
            messages=[ChatMessage(role="user", content="hello")],
            tool_choice="required",
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
            adapter.calls[1]["input_payload"],
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
        self.assertEqual(adapter.calls[1]["tool_choice"], "auto")

    def test_chat_completions_function_tool_choice_is_normalized_for_responses_api(self) -> None:
        adapter = CapturingAdapter()
        service = ChatService(
            adapter,
            DummyToolExecutor(),
            DummyAuditLogger(),
            default_model="gpt-5.4-mini",
        )

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


if __name__ == "__main__":
    unittest.main()
