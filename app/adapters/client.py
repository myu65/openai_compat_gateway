from __future__ import annotations

import httpx
from openai import AsyncOpenAI, OpenAI


def _timeout(
    *,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    write_timeout_seconds: float,
    pool_timeout_seconds: float,
) -> httpx.Timeout:
    return httpx.Timeout(
        timeout=read_timeout_seconds,
        connect=connect_timeout_seconds,
        read=read_timeout_seconds,
        write=write_timeout_seconds,
        pool=pool_timeout_seconds,
    )


def _limits(*, max_connections: int, max_keepalive_connections: int) -> httpx.Limits:
    return httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
    )


def create_openai_client(
    *,
    api_key: str | None,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    write_timeout_seconds: float,
    pool_timeout_seconds: float,
    max_retries: int,
    max_connections: int = 64,
    max_keepalive_connections: int = 0,
) -> OpenAI:
    """Create a sync OpenAI client with bounded timeouts and connection pooling."""
    timeout = _timeout(
        connect_timeout_seconds=connect_timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
        write_timeout_seconds=write_timeout_seconds,
        pool_timeout_seconds=pool_timeout_seconds,
    )
    http_client = httpx.Client(
        timeout=timeout,
        limits=_limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        ),
        follow_redirects=True,
    )
    return OpenAI(
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        http_client=http_client,
    )


def create_async_openai_client(
    *,
    api_key: str | None,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    write_timeout_seconds: float,
    pool_timeout_seconds: float,
    max_retries: int,
    max_connections: int = 64,
    max_keepalive_connections: int = 0,
) -> AsyncOpenAI:
    """Create the production async OpenAI client with bounded connection resources."""
    timeout = _timeout(
        connect_timeout_seconds=connect_timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
        write_timeout_seconds=write_timeout_seconds,
        pool_timeout_seconds=pool_timeout_seconds,
    )
    http_client = httpx.AsyncClient(
        timeout=timeout,
        limits=_limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        ),
        follow_redirects=True,
    )
    return AsyncOpenAI(
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        http_client=http_client,
    )
