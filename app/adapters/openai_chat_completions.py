from __future__ import annotations

from openai import OpenAI


class OpenAIChatCompletionsAdapter:
    """Native pass-through used when Responses translation is unnecessary."""

    def __init__(self, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key)

    def create_completion(self, payload: dict, *, stream: bool = False):
        request = dict(payload)
        request["store"] = False
        request["stream"] = stream
        return self.client.chat.completions.create(**request)
