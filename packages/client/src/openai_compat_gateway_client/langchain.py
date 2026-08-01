"""LangChain client that preserves the gateway's lossless state extension."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


def _merge_state(*states: Any) -> dict[str, Any] | None:
    """Merge top-level observability data with per-message replay state."""

    merged: dict[str, Any] = {}
    for state in states:
        if isinstance(state, Mapping):
            merged.update(state)
    return merged or None


def _validate_private_api() -> None:
    """Fail clearly if an unsupported LangChain private API is forced in."""

    expected = {
        "_get_request_payload": ["self", "input_", "stop", "kwargs"],
        "_create_chat_result": ["self", "response", "generation_info"],
        "_convert_chunk_to_generation_chunk": [
            "self",
            "chunk",
            "default_chunk_class",
            "base_generation_info",
        ],
    }
    for method_name, expected_parameters in expected.items():
        method = getattr(ChatOpenAI, method_name, None)
        if method is None:
            raise ImportError(f"Unsupported langchain-openai: ChatOpenAI.{method_name} is missing")
        actual_parameters = list(inspect.signature(method).parameters)
        if actual_parameters != expected_parameters:
            raise ImportError(
                f"Unsupported langchain-openai private API: ChatOpenAI.{method_name}{inspect.signature(method)}"
            )


_validate_private_api()


class ChatOpenAICompat(ChatOpenAI):
    """Round-trip the gateway's lossless Responses state through LangChain.

    Stock ``langchain-openai`` discards unknown Chat Completions fields. This
    subclass preserves ``x_openai`` in ``AIMessage.additional_kwargs`` and
    sends it back on the corresponding assistant message.
    """

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        source_messages = self._convert_input(input_).to_messages()
        for source, encoded in zip(source_messages, payload.get("messages", []), strict=False):
            if isinstance(source, AIMessage) and source.additional_kwargs.get("x_openai"):
                encoded["x_openai"] = source.additional_kwargs["x_openai"]
        return payload

    def _create_chat_result(self, response: Any, generation_info: dict | None = None):
        response_dict = response if isinstance(response, Mapping) else response.model_dump(warnings=False)
        top_level_state = response_dict.get("x_openai")
        message_state = None
        choices = response_dict.get("choices") or []
        if choices:
            message_state = choices[0].get("message", {}).get("x_openai")
        state = _merge_state(top_level_state, message_state)
        result = super()._create_chat_result(response, generation_info=generation_info)
        if state and result.generations:
            result.generations[0].message.additional_kwargs["x_openai"] = state
        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ):
        generation = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if generation is None:
            return None
        top_level_state = chunk.get("x_openai")
        delta_state = None
        choices = chunk.get("choices") or []
        if choices:
            delta_state = choices[0].get("delta", {}).get("x_openai")
        state = _merge_state(top_level_state, delta_state)
        if state:
            generation.message.additional_kwargs["x_openai"] = state
        return generation
