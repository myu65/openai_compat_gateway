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
        metadata=None,
        moderation=None,
        prompt_cache_key=None,
        prompt_cache_options=None,
        prompt_cache_retention=None,
        safety_identifier=None,
        user=None,
        stream_options=None,
        stream: bool = False,
    ):
        kwargs = {
            "model": model,
            "input": input_payload,
            "store": False,
            "stream": stream,
        }
        optional = {
            "tools": tools or None,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "include": include or None,
            "reasoning": reasoning,
            "max_output_tokens": max_output_tokens,
            "top_p": top_p,
            "text": text,
            "parallel_tool_calls": parallel_tool_calls,
            "service_tier": service_tier,
            "metadata": metadata,
            "moderation": moderation,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_options": prompt_cache_options,
            "prompt_cache_retention": prompt_cache_retention,
            "safety_identifier": safety_identifier,
            "user": user,
            "stream_options": stream_options,
        }
        kwargs.update({key: value for key, value in optional.items() if value is not None})
        return self.client.responses.create(**kwargs)
