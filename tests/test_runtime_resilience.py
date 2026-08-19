from __future__ import annotations

import json

import httpx
from openai import APITimeoutError

from app.adapters.client import create_openai_client
from app.api.chat_completions import _error_response, _timeout_safe_sse


def test_openai_client_has_explicit_granular_timeouts() -> None:
    client = create_openai_client(
        api_key="test-key",
        connect_timeout_seconds=1.5,
        read_timeout_seconds=901.0,
        write_timeout_seconds=12.0,
        pool_timeout_seconds=3.0,
        max_retries=0,
    )

    try:
        assert client.timeout.connect == 1.5
        assert client.timeout.read == 901.0
        assert client.timeout.write == 12.0
        assert client.timeout.pool == 3.0
        assert client.max_retries == 0
    finally:
        client.close()


def test_upstream_timeout_is_returned_as_gateway_timeout() -> None:
    exc = APITimeoutError(httpx.Request("POST", "https://api.openai.com/v1/responses"))

    response = _error_response(exc)
    body = json.loads(response.body)

    assert response.status_code == 504
    assert body["error"]["type"] == "api_error"
    assert body["error"]["code"] == "upstream_timeout"


def test_stream_timeout_emits_sse_error_and_done() -> None:
    def timed_out_stream():
        yield 'data: {"choices":[]}\n\n'
        raise APITimeoutError(httpx.Request("POST", "https://api.openai.com/v1/responses"))

    chunks = list(_timeout_safe_sse(timed_out_stream()))

    assert chunks[0] == 'data: {"choices":[]}\n\n'
    assert '"code": "upstream_timeout"' in chunks[1]
    assert chunks[2] == "data: [DONE]\n\n"
