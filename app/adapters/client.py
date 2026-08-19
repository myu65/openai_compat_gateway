from __future__ import annotations

import httpx
from openai import OpenAI


def create_openai_client(
    *,
    api_key: str | None,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    write_timeout_seconds: float,
    pool_timeout_seconds: float,
    max_retries: int,
) -> OpenAI:
    """Create an OpenAI client with explicit timeouts for every network phase."""
    timeout = httpx.Timeout(
        timeout=read_timeout_seconds,
        connect=connect_timeout_seconds,
        read=read_timeout_seconds,
        write=write_timeout_seconds,
        pool=pool_timeout_seconds,
    )
    return OpenAI(
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )
