from __future__ import annotations

from app.adapters.client import create_openai_client


class OpenAIChatCompletionsAdapter:
    """Sync adapter retained for direct use and existing compatibility tests."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 900.0,
        write_timeout_seconds: float = 30.0,
        pool_timeout_seconds: float = 10.0,
        max_retries: int = 0,
        max_connections: int = 64,
        max_keepalive_connections: int = 0,
    ):
        self.client = create_openai_client(
            api_key=api_key,
            connect_timeout_seconds=connect_timeout_seconds,
            read_timeout_seconds=read_timeout_seconds,
            write_timeout_seconds=write_timeout_seconds,
            pool_timeout_seconds=pool_timeout_seconds,
            max_retries=max_retries,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )

    def create_completion(self, payload: dict, *, stream: bool = False):
        request = dict(payload)
        request["store"] = False
        request["stream"] = stream
        return self.client.chat.completions.create(**request)


class AsyncOpenAIChatCompletionsAdapter:
    """Async pass-through used by the FastAPI runtime."""

    def __init__(self, client):
        self.client = client

    async def create_completion(self, payload: dict, *, stream: bool = False):
        request = dict(payload)
        request["store"] = False
        request["stream"] = stream
        return await self.client.chat.completions.create(**request)
