from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from app.mappers.stream_mapper import map_stream_events
from app.schemas.internal import BridgeExecution


class StreamMapperTests(unittest.TestCase):
    def test_stream_completed_emits_legacy_steps_for_bridged_web_search(self) -> None:
        events = [
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="web_search_call",
                    id="ws_123",
                    status="completed",
                    action=SimpleNamespace(
                        query="tokyo weather",
                        sources=[SimpleNamespace(url="https://example.com", title="Example")],
                    ),
                ),
            ),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text="sunny",
                            annotations=[
                                SimpleNamespace(
                                    type="url_citation",
                                    url="https://example.com/detail",
                                    title="Detail",
                                )
                            ],
                        )
                    ],
                ),
            ),
            SimpleNamespace(type="response.completed"),
        ]

        chunks = list(
            map_stream_events(
                events,
                "gpt-5.4-mini",
                bridge_executions=[
                    BridgeExecution(
                        requested_tool_name="web_search",
                        display_tool_name="web_search",
                        builtin_tool_type="web_search",
                    )
                ],
            )
        )

        final_payload = json.loads(chunks[-2][len("data: ") :].strip())
        legacy_steps = final_payload["x_openai"]["legacy_steps"]
        self.assertEqual(legacy_steps[0]["tool_calls"][0]["function"]["name"], "web_search")
        self.assertEqual(legacy_steps[1]["name"], "web_search")
        self.assertEqual(final_payload["x_openai"]["assistant_text"], "sunny")
        self.assertEqual(len(final_payload["x_openai"]["citations"]), 2)

    def test_stream_function_call_finishes_with_tool_calls(self) -> None:
        events = [
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

        chunks = list(map_stream_events(events, "gpt-5.4-mini"))

        final_payload = json.loads(chunks[-2][len("data: ") :].strip())
        self.assertEqual(final_payload["choices"][0]["finish_reason"], "tool_calls")


if __name__ == "__main__":
    unittest.main()
