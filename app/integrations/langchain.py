from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from langchain_core.messages import AIMessage
    from langchain_openai import ChatOpenAI
except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
    raise ImportError("Install openai-compat-gateway[langchain] to use ChatOpenAICompat") from exc


class ChatOpenAICompat(ChatOpenAI):
    """ChatOpenAI that round-trips this gateway's lossless Responses state.

    Stock LangChain intentionally discards unknown Chat Completions fields. This
    small subclass preserves ``x_openai`` in ``AIMessage.additional_kwargs`` and
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
        state = response_dict.get("x_openai")
        choices = response_dict.get("choices") or []
        if choices:
            state = choices[0].get("message", {}).get("x_openai") or state
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
        state = chunk.get("x_openai")
        choices = chunk.get("choices") or []
        if choices:
            state = choices[0].get("delta", {}).get("x_openai") or state
        if state:
            generation.message.additional_kwargs["x_openai"] = state
        return generation
