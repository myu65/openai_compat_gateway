from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.adapters.client import _limits
from app.api.chat_completions import InflightLimiter, _guarded_async_sse, _inflight, _sse_headers
from app.mappers.async_stream_mapper import map_async_stream_events
from app.schemas.compat import ChatCompletionsRequest, ChatMessage
from app.services.async_chat_service import AsyncChatService


class ClosingAsyncStream:
    def __init__(self, events, *, delay_seconds: float = 0):
        self.events = list(events)
        self.delay_seconds = delay_seconds
        self.closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self.events:
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            yield event

    async def close(self):
        self.closed = True


class AsyncCapturingAdapter:
    def __init__(self, response, *, delay_seconds: float = 0):
        self.response = response
        self.delay_seconds = delay_seconds
        self.calls = []

    async def create_response(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        return self.response


class DummyAuditLogger:
    def __init__(self):
        self.logged = []

    def log_chat(self, req, normalized):
        self.logged.append((req, normalized))
        return {}


class NullToolExecutor:
    def has(self, _name: str) -> bool:
        return False


def test_connection_pool_limits_are_explicit_and_bounded() -> None:
    limits = _limits(max_connections=17, max_keepalive_connections=0)
    assert limits.max_connections == 17
    assert limits.max_keepalive_connections == 0


def test_inflight_limiter_fails_fast_and_recovers() -> None:
    limiter = InflightLimiter(1)
    assert limiter.try_acquire() is True
    assert limiter.active == 1
    assert limiter.try_acquire() is False
    limiter.release()
    assert limiter.active == 0
    assert limiter.try_acquire() is True
    limiter.release()


def test_stream_headers_disable_keepalive_by_default() -> None:
    headers = _sse_headers("req-1")
    assert headers["X-Request-Id"] == "req-1"
    assert headers["Connection"] == "close"


def test_async_stream_mapper_closes_upstream_on_consumer_close() -> None:
    async def run() -> None:
        upstream = ClosingAsyncStream([SimpleNamespace(type="response.output_text.delta", delta="hello")])
        mapped = map_async_stream_events(upstream, "test-model")
        first = await anext(mapped)
        assert '"role": "assistant"' in first
        await mapped.aclose()
        assert upstream.closed is True

    asyncio.run(run())


def test_total_deadline_ends_stream_and_releases_capacity() -> None:
    async def stalled_stream():
        yield 'data: {"choices":[]}\n\n'
        await asyncio.Event().wait()

    async def run() -> None:
        before = _inflight.active
        assert _inflight.try_acquire() is True
        chunks = []
        async for chunk in _guarded_async_sse(
            stalled_stream(),
            request_id="deadline-test",
            mode="responses",
            deadline_seconds=0.01,
        ):
            chunks.append(chunk)
        assert _inflight.active == before
        assert chunks[0] == 'data: {"choices":[]}\n\n'
        assert json.loads(chunks[1][len("data: ") :])["error"]["code"] == "upstream_timeout"
        assert chunks[2] == "data: [DONE]\n\n"

    asyncio.run(run())


def test_async_service_awaits_adapter_without_sync_threadpool() -> None:
    async def run() -> None:
        adapter = AsyncCapturingAdapter(SimpleNamespace(output=[], usage={"total_tokens": 1}))
        audit = DummyAuditLogger()
        service = AsyncChatService(
            adapter=adapter,
            tool_executor=NullToolExecutor(),
            audit_logger=audit,
            default_model="test-model",
        )
        req = ChatCompletionsRequest(messages=[ChatMessage(role="user", content="hello")])

        normalized = await service.run_nonstream_async(req)

        assert normalized.usage["total_tokens"] == 1
        assert adapter.calls[0]["stream"] is False
        assert audit.logged

    asyncio.run(run())


def test_stream_deadline_is_cumulative_across_creation_and_iteration() -> None:
    async def run() -> None:
        upstream = ClosingAsyncStream(
            [SimpleNamespace(type="response.output_text.delta", delta="too late")],
            delay_seconds=0.06,
        )
        adapter = AsyncCapturingAdapter(upstream, delay_seconds=0.06)
        service = AsyncChatService(
            adapter=adapter,
            tool_executor=NullToolExecutor(),
            audit_logger=DummyAuditLogger(),
            default_model="test-model",
            request_deadline_seconds=0.1,
        )
        req = ChatCompletionsRequest(
            messages=[ChatMessage(role="user", content="hello")],
            stream=True,
            reasoning_effort="medium",
            tools=[],
            x_openai={"mode": "responses"},
        )

        stream = await service.run_stream_async(req)
        with pytest.raises(TimeoutError):
            async for _chunk in stream:
                pass
        assert upstream.closed is True

    asyncio.run(run())
