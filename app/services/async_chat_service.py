from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any

from app.config import settings
from app.mappers.async_stream_mapper import map_async_stream_events
from app.mappers.response_mapper import normalize_final_response
from app.services.chat_service import ChatService


class _DeadlineAsyncIterable:
    """Apply one absolute deadline across stream creation and later iteration."""

    def __init__(self, stream: Any, deadline_at: float):
        self.stream = stream
        self.deadline_at = deadline_at
        self._closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        remaining = self.deadline_at - time.monotonic()
        if remaining <= 0:
            await self.aclose()
            raise TimeoutError
        try:
            async with asyncio.timeout(remaining):
                async for item in self.stream:
                    yield item
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.stream, "aclose", None)
        if close is None:
            close = getattr(self.stream, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result


class AsyncChatService(ChatService):
    """Async runtime facade that reuses ChatService's compatibility mapping logic."""

    def __init__(self, *args, request_deadline_seconds: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_deadline_seconds = (
            settings.openai_request_deadline_seconds if request_deadline_seconds is None else request_deadline_seconds
        )

    def _deadline_at(self) -> float:
        return time.monotonic() + self.request_deadline_seconds

    @staticmethod
    def _remaining(deadline_at: float) -> float:
        remaining = deadline_at - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        return remaining

    async def run_native_nonstream_async(self, req):
        return await self.native_adapter.create_completion(self._native_payload(req), stream=False)

    async def run_native_stream_async(self, req):
        deadline_at = self._deadline_at()
        async with asyncio.timeout(self._remaining(deadline_at)):
            stream = await self.native_adapter.create_completion(self._native_payload(req), stream=True)
        return _DeadlineAsyncIterable(stream, deadline_at)

    async def run_nonstream_async(self, req):
        model, input_payload, tools, include, bridge_requests = self._prepare_request(req)
        resp = await self.adapter.create_response(
            model=model,
            input_payload=input_payload,
            tools=tools,
            tool_choice=self._normalize_tool_choice(req.tool_choice),
            include=include,
            stream=False,
            **self._shared_responses_kwargs(req),
        )
        normalized = normalize_final_response(resp, bridge_executions=bridge_requests)
        self.audit_logger.log_chat(req, normalized)
        return normalized

    async def run_stream_async(self, req):
        deadline_at = self._deadline_at()
        model, input_payload, tools, include, bridge_requests = self._prepare_request(req)
        async with asyncio.timeout(self._remaining(deadline_at)):
            openai_stream = await self.adapter.create_response(
                model=model,
                input_payload=input_payload,
                tools=tools,
                tool_choice=self._normalize_tool_choice(req.tool_choice),
                include=include,
                stream=True,
                **self._shared_responses_kwargs(req),
            )
        mapped = map_async_stream_events(
            openai_stream,
            model,
            bridge_executions=bridge_requests,
            include_usage=bool(req.stream_options and req.stream_options.get("include_usage")),
        )
        return _DeadlineAsyncIterable(mapped, deadline_at)
