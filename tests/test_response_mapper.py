from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.mappers.response_mapper import _normalize_usage, normalize_final_response


class NormalizeFinalResponseTests(unittest.TestCase):
    def test_usage_model_dump_is_normalized_to_dict(self) -> None:
        class UsageObject:
            def model_dump(self):
                return {"input_tokens": 12, "output_tokens": 34, "total_tokens": 46}

        normalized = _normalize_usage(UsageObject())

        self.assertEqual(
            normalized,
            {"input_tokens": 12, "output_tokens": 34, "total_tokens": 46},
        )

    def test_web_search_without_results_still_normalizes(self) -> None:
        resp = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="web_search_call",
                    id="ws_123",
                    status="completed",
                    action=SimpleNamespace(
                        query="weather",
                        sources=[
                            SimpleNamespace(url="https://example.com", title="Example"),
                        ],
                    ),
                ),
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text="answer",
                            annotations=[
                                SimpleNamespace(
                                    type="url_citation",
                                    url="https://example.org/detail",
                                    title="Example Detail",
                                )
                            ],
                        )
                    ],
                ),
            ],
            usage=SimpleNamespace(input_tokens=3, output_tokens=7, total_tokens=10),
        )

        normalized = normalize_final_response(resp)

        self.assertEqual(normalized.assistant_text, "answer")
        self.assertEqual(len(normalized.citations), 2)
        self.assertEqual(normalized.citations[0].url, "https://example.com")
        self.assertEqual(normalized.citations[1].url, "https://example.org/detail")
        self.assertEqual(normalized.citations[1].title, "Example Detail")
        self.assertEqual(normalized.builtin_tool_events[0].payload["sources_count"], 1)
        self.assertEqual(
            normalized.usage,
            {"input_tokens": 3, "output_tokens": 7, "total_tokens": 10},
        )


if __name__ == "__main__":
    unittest.main()
