from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.mappers.legacy_log_mapper import to_legacy_log_steps


class LegacyLogMapperTests(unittest.TestCase):
    def test_message_annotations_are_exposed_as_citations(self) -> None:
        resp = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text="answer",
                            annotations=[
                                SimpleNamespace(
                                    type="url_citation",
                                    url="https://example.com",
                                    title="Example",
                                )
                            ],
                        )
                    ],
                )
            ]
        )

        steps = to_legacy_log_steps(resp)

        self.assertEqual(steps[0]["content"], "answer")
        self.assertEqual(
            steps[0]["citations"],
            [{"title": "Example", "url": "https://example.com"}],
        )


if __name__ == "__main__":
    unittest.main()
