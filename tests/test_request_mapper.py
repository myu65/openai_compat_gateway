from __future__ import annotations

import unittest

from app.mappers.request_mapper import build_include_list, to_responses_input
from app.schemas.compat import BuiltinToolsConfig


class BuildIncludeListTests(unittest.TestCase):
    def test_web_search_defaults_to_sources_only(self) -> None:
        include = build_include_list(BuiltinToolsConfig(web_search=True))
        self.assertEqual(include, ["web_search_call.action.sources"])

    def test_web_search_results_can_be_enabled(self) -> None:
        include = build_include_list(
            BuiltinToolsConfig(web_search=True),
            include_web_search_results=True,
        )
        self.assertEqual(
            include,
            ["web_search_call.action.sources", "web_search_call.results"],
        )

    def test_other_builtin_includes_remain_enabled(self) -> None:
        include = build_include_list(
            BuiltinToolsConfig(
                web_search=True,
                file_search={"vector_store_ids": ["vs_123"]},
                code_interpreter=True,
            )
        )
        self.assertEqual(
            include,
            [
                "web_search_call.action.sources",
                "file_search_call.results",
                "code_interpreter_call.outputs",
            ],
        )


class ToResponsesInputTests(unittest.TestCase):
    def test_assistant_tool_calls_and_tool_outputs_are_replayed_for_followup(self) -> None:
        items = to_responses_input(
            [
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
                    "content": {"role": "admin"},
                },
            ]
        )

        self.assertEqual(
            items,
            [
                {"role": "user", "content": "u1を見て"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup_profile",
                    "arguments": '{"user_id":"u1"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '{"role": "admin"}',
                },
            ],
        )

    def test_assistant_message_with_text_is_preserved_before_tool_calls(self) -> None:
        items = to_responses_input(
            [
                {
                    "role": "assistant",
                    "content": "まず確認します",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup_profile",
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            ]
        )

        self.assertEqual(
            items,
            [
                {"role": "assistant", "content": "まず確認します"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup_profile",
                    "arguments": "{}",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
