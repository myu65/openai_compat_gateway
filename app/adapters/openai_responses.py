from __future__ import annotations

from openai import OpenAI


class OpenAIResponsesAdapter:
    def __init__(self, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key)

    def create_response(
        self,
        *,
        model: str,
        input_payload,
        tools=None,
        tool_choice=None,
        temperature=None,
        include=None,
        stream: bool = False,
    ):
        kwargs = {
            "model": model,
            "input": input_payload,
            "store": False,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if temperature is not None:
            kwargs["temperature"] = temperature
        if include:
            kwargs["include"] = include
        return self.client.responses.create(**kwargs)
