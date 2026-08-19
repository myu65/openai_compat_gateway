from __future__ import annotations

import unittest

from app.mappers.request_mapper import build_include_list, to_responses_input
from app.schemas.compat import BuiltinToolsConfig


class BuildIncludeListTests(unittest.TestCase):
    def test_web_search_defaults_to_sources_only(self) -> None:
        include = build_include_list(BuiltinToolsConfig(web_search=True))
        self.assertEqual(
            include,
            ["web_search_call.action.sources", "reasoning.encrypted_content"],
        )

    def test_web_search_results_can_be_enabled(self) -> None:
        include = build_include_list(
            BuiltinToolsConfig(web_search=True),
            include_web_search_results=True,
        )
        self.assertEqual(
            include,
            [
                "web_search_call.action.sources",
                "web_search_call.results",
                "reasoning.encrypted_content",
            ],
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
                "reasoning.encrypted_content",
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

    def test_chat_content_parts_are_translated_to_responses_shapes(self) -> None:
        items = to_responses_input(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "この画像とファイルを見て",
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,AAAA",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "file",
                            "file": {
                                "file_id": "file_123",
                                "filename": "sample.pdf",
                            },
                        },
                    ],
                }
            ]
        )

        self.assertEqual(
            items,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "この画像とファイルを見て",
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        },
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AAAA",
                            "detail": "high",
                        },
                        {
                            "type": "input_file",
                            "file_id": "file_123",
                            "filename": "sample.pdf",
                        },
                    ],
                }
            ],
        )

    def test_image_url_string_is_accepted_defensively(self) -> None:
        items = to_responses_input(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/image.png",
                        }
                    ],
                }
            ]
        )

        self.assertEqual(
            items,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/image.png",
                        }
                    ],
                }
            ],
        )

    def test_tool_output_text_parts_are_translated_instead_of_json_encoded(self) -> None:
        items = to_responses_input(
            [
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "text", "text": "second"},
                    ],
                }
            ]
        )

        self.assertEqual(
            items,
            [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "input_text", "text": "first"},
                        {"type": "input_text", "text": "second"},
                    ],
                }
            ],
        )

    def test_assistant_refusal_part_is_replayed_as_text(self) -> None:
        items = to_responses_input(
            [
                {
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "できません"}],
                }
            ]
        )

        self.assertEqual(
            items,
            [
                {
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": "できません"}],
                }
            ],
        )

    def test_assistant_top_level_refusal_is_replayed_as_text(self) -> None:
        items = to_responses_input(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "refusal": "対応できません",
                }
            ]
        )

        self.assertEqual(
            items,
            [
                {
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": "対応できません"}],
                }
            ],
        )

    def test_input_audio_fails_clearly_in_responses_translation(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support Chat Completions input_audio"):
            to_responses_input(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": "AAAA", "format": "mp3"},
                            }
                        ],
                    }
                ]
            )

    def test_unknown_chat_content_part_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported Chat Completions content part"):
            to_responses_input(
                [
                    {
                        "role": "user",
                        "content": [{"type": "future_type", "value": "x"}],
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
