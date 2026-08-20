from __future__ import annotations

from app.mappers.async_stream_mapper import map_async_stream_events
from app.mappers.response_mapper import normalize_final_response
from app.services.chat_service import ChatService


class AsyncChatService(ChatService):
    """Async runtime facade that reuses ChatService's compatibility mapping logic."""

    async def run_native_nonstream_async(self, req):
        return await self.native_adapter.create_completion(self._native_payload(req), stream=False)

    async def run_native_stream_async(self, req):
        return await self.native_adapter.create_completion(self._native_payload(req), stream=True)

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
        model, input_payload, tools, include, bridge_requests = self._prepare_request(req)
        openai_stream = await self.adapter.create_response(
            model=model,
            input_payload=input_payload,
            tools=tools,
            tool_choice=self._normalize_tool_choice(req.tool_choice),
            include=include,
            stream=True,
            **self._shared_responses_kwargs(req),
        )
        return map_async_stream_events(
            openai_stream,
            model,
            bridge_executions=bridge_requests,
            include_usage=bool(req.stream_options and req.stream_options.get("include_usage")),
        )
