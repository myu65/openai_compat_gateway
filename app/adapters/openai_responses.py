from __future__ import annotations

from app.adapters.client import create_openai_client


class OpenAIResponsesAdapter:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 900.0,
        write_timeout_seconds: float = 30.0,
        pool_timeout_seconds: float = 10.0,
        max_retries: int = 0,
    ):
        self.client = create_openai_client(
            api_key=api_key,
            connect_timeout_seconds=connect_timeout_seconds,
            read_timeout_seconds=read_timeout_seconds,
            write_timeout_seconds=write_timeout_seconds,
            pool_timeout_seconds=pool_timeout_seconds,
            max_retries=max_retries,
        )

    def create_response(
        self,
        *,
        model: str,
        input_payload,
        tools=None,
        tool_choice=None,
        temperature=None,
        include=None,
        reasoning=None,
        max_output_tokens=None,
        top_p=None,
        text=None,
        parallel_tool_calls=None,
        service_tier=None,
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
        if reasoning is not None:
            kwargs["reasoning"] = reasoning
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        if top_p is not None:
            kwargs["top_p"] = top_p
        if text is not None:
            kwargs["text"] = text
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        if service_tier is not None:
            kwargs["service_tier"] = service_tier
        return self.client.responses.create(**kwargs)
