from __future__ import annotations

import json
import os
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from langchain_core.messages import HumanMessage, ToolMessage
from openai_compat_gateway_client import ChatOpenAICompat

from app.adapters.openai_responses import OpenAIResponsesAdapter
from app.api.chat_completions import get_service
from app.config import settings
from app.main import app
from app.services.chat_service import ChatService


class DummyAuditLogger:
    def log_chat(self, _req, _normalized) -> dict:
        return {}


class NullToolExecutor:
    def has(self, _name: str) -> bool:
        return False


@pytest.mark.e2e
@pytest.mark.skipif(
    not os.getenv("RUN_OPENAI_E2E") or not settings.openai_api_key,
    reason="requires RUN_OPENAI_E2E=1 and OPENAI_API_KEY",
)
def test_real_openai_langchain_reasoning_custom_tool_round_trip() -> None:
    service = ChatService(
        adapter=OpenAIResponsesAdapter(api_key=settings.openai_api_key),
        tool_executor=NullToolExecutor(),
        audit_logger=DummyAuditLogger(),
        default_model=settings.openai_model_default,
    )
    app.dependency_overrides[get_service] = lambda: service

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()

    llm: ChatOpenAICompat | None = None
    try:
        for _ in range(500):
            if server.started:
                break
            if not thread.is_alive():
                raise RuntimeError("local E2E gateway stopped before startup")
            time.sleep(0.01)
        else:
            raise RuntimeError("local E2E gateway did not start")

        llm = ChatOpenAICompat(
            model=settings.openai_model_default,
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="local-gateway-e2e-key",
            reasoning_effort="medium",
            http_client=httpx.Client(trust_env=False),
            http_async_client=httpx.AsyncClient(trust_env=False),
        )
        tool = {
            "type": "function",
            "function": {
                "name": "lookup_order_status",
                "description": "Return the shipment status for an order id.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }
        bound = llm.bind_tools([tool], tool_choice="lookup_order_status")
        user = HumanMessage(
            content=("Call lookup_order_status exactly once for ord_314, then report the tracking code and status.")
        )
        assistant = bound.invoke([user])

        assert assistant.tool_calls
        assert assistant.additional_kwargs["x_openai"]["response_items"]
        call = assistant.tool_calls[0]
        assert call["name"] == "lookup_order_status"
        assert call["args"]["order_id"] == "ord_314"

        tool_result = ToolMessage(
            content=json.dumps(
                {
                    "order_id": "ord_314",
                    "status": "shipped",
                    "tracking_code": "ZX-31415",
                }
            ),
            tool_call_id=call["id"],
        )
        final = llm.bind_tools([tool], tool_choice="auto").invoke([user, assistant, tool_result])

        assert "ZX-31415" in str(final.content)
        assert "shipped" in str(final.content).lower()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        app.dependency_overrides.clear()
