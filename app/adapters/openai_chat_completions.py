from __future__ import annotations

from app.adapters.client import create_openai_client


class OpenAIChatCompletionsAdapter:
    """Native pass-through used when Responses translation is unnecessary."""

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

    def create_completion(self, payload: dict, *, stream: bool = False):
        request = dict(payload)
        request["store"] = False
        request["stream"] = stream
        return self.client.chat.completions.create(**request)
