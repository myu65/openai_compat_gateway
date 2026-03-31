from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.mappers.response_mapper import normalize_final_response


class NormalizeFinalResponseTests(unittest.TestCase):
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
                    content=[SimpleNamespace(type="output_text", text="answer")],
                ),
            ],
            usage={"total_tokens": 10},
        )

        normalized = normalize_final_response(resp)

        self.assertEqual(normalized.assistant_text, "answer")
        self.assertEqual(len(normalized.citations), 1)
        self.assertEqual(normalized.citations[0].url, "https://example.com")
        self.assertEqual(normalized.builtin_tool_events[0].payload["sources_count"], 1)


if __name__ == "__main__":
    unittest.main()
