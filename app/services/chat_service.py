from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from app.mappers.request_mapper import (
    build_include_list,
    merge_builtin_tools,
    to_responses_custom_tools,
    to_responses_input,
    to_responses_text_config,
)
from app.mappers.response_mapper import normalize_final_response
from app.mappers.stream_mapper import map_stream_events
from app.schemas.internal import BridgeExecution
from app.tools.registry import extract_bridge_query, find_bridge_for_tool_name


class ChatService:
    def __init__(
        self,
        adapter,
        tool_executor,
        audit_logger,
        default_model: str,
        include_web_search_results: bool = False,
        native_adapter=None,
    ):
        self.adapter = adapter
        self.tool_executor = tool_executor
        self.audit_logger = audit_logger
        self.default_model = default_model
        self.include_web_search_results = include_web_search_results
        self.native_adapter = native_adapter

    def select_mode(self, req) -> str:
        requested = req.x_openai.mode if req.x_openai else "auto"
        if requested != "auto":
            if requested == "chat_completions" and self.native_adapter is None:
                raise ValueError("Native Chat Completions mode is not configured")
            return requested
        if self.native_adapter is None:
            return "responses"
        has_state = bool(req.x_openai and req.x_openai.input_items) or any(
            bool(message.x_openai and message.x_openai.response_items) for message in req.messages
        )
        has_bridge = any(find_bridge_for_tool_name(tool.function.name) for tool in (req.tools or []))
        if req.x_builtin_tools or has_state or has_bridge or (req.reasoning_effort is not None and req.tools):
            return "responses"
        return "chat_completions"

    def _native_payload(self, req) -> dict[str, Any]:
        payload = req.model_dump(exclude_none=True)
        payload["model"] = req.model or self.default_model
        payload.pop("x_builtin_tools", None)
        payload.pop("x_openai", None)
        if not req.tools:
            payload.pop("tool_choice", None)
        for message in payload.get("messages", []):
            message.pop("x_openai", None)
        return payload

    def run_native_nonstream(self, req):
        return self.native_adapter.create_completion(self._native_payload(req), stream=False)

    def run_native_stream(self, req):
        return self.native_adapter.create_completion(self._native_payload(req), stream=True)

    def _validate_responses_compatibility(self, req) -> None:
        conflicts: list[str] = []
        if req.n != 1:
            conflicts.append("n")
        if req.stop is not None:
            conflicts.append("stop")
        if any(message.role == "function" for message in req.messages):
            conflicts.append("legacy role=function messages")
        if conflicts:
            joined = ", ".join(conflicts)
            raise ValueError(f"Responses translation cannot preserve {joined}; use x_openai.mode='chat_completions'")

    def _normalize_tool_choice(self, tool_choice):
        if not isinstance(tool_choice, dict):
            return tool_choice
        if tool_choice.get("type") != "function":
            return tool_choice
        function = tool_choice.get("function")
        if not isinstance(function, dict):
            return tool_choice
        name = function.get("name")
        if not name:
            return tool_choice
        return {"type": "function", "name": name}

    def _tool_choice_for_followup(self, original_tool_choice):
        # After the model emits a custom function call, the follow-up request
        # should let the model produce the final answer from tool outputs rather
        # than forcing another tool invocation.
        if original_tool_choice in ("required", "auto"):
            return "auto"
        return None

    def _prepare_request(self, req):
        self._validate_responses_compatibility(req)
        model = req.model or self.default_model
        builtin_cfg = deepcopy(req.x_builtin_tools)
        custom_tools = list(req.tools or [])
        bridge_requests: list[BridgeExecution] = []
        bridge_messages: list[dict[str, Any]] = []
        forwarded_messages = []
        bridged_calls: dict[str, tuple[Any, str, dict[str, Any]]] = {}

        for tool in custom_tools:
            spec = find_bridge_for_tool_name(tool.function.name)
            if not spec:
                continue
            bridge_requests.append(
                BridgeExecution(
                    requested_tool_name=tool.function.name,
                    display_tool_name=tool.function.name,
                    builtin_tool_type=spec.builtin_tool_type,
                )
            )
            if spec.builtin_tool_type == "web_search":
                if builtin_cfg is None:
                    from app.schemas.compat import BuiltinToolsConfig

                    builtin_cfg = BuiltinToolsConfig(web_search=True)
                else:
                    builtin_cfg.web_search = True
            elif spec.builtin_tool_type == "file_search":
                if builtin_cfg is None:
                    from app.schemas.compat import BuiltinToolsConfig

                    builtin_cfg = BuiltinToolsConfig(file_search={})
                else:
                    builtin_cfg.file_search = builtin_cfg.file_search or {}

        for message in req.messages:
            if message.role != "assistant":
                continue
            for tool_call in message.tool_calls or []:
                function = tool_call.get("function") or {}
                name = function.get("name")
                spec = find_bridge_for_tool_name(name) if name else None
                call_id = tool_call.get("id")
                if not spec or not call_id:
                    continue
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except (TypeError, json.JSONDecodeError):
                    arguments = {}
                bridged_calls[call_id] = (spec, name, arguments if isinstance(arguments, dict) else {})

        # Bridge selected custom tools based on prior legacy tool calls already in messages.
        for message in req.messages:
            if message.role == "assistant" and message.tool_calls:
                encoded = message.model_dump()
                encoded["tool_calls"] = [
                    tool_call for tool_call in message.tool_calls if tool_call.get("id") not in bridged_calls
                ]
                if encoded["tool_calls"] or encoded.get("content") not in (None, "", []):
                    forwarded_messages.append(encoded)
                continue

            if message.role == "tool":
                prior_bridge = bridged_calls.get(message.tool_call_id or "")
                spec = (
                    prior_bridge[0]
                    if prior_bridge
                    else find_bridge_for_tool_name(message.name)
                    if message.name
                    else None
                )
                if not spec:
                    forwarded_messages.append(message.model_dump())
                    continue
                try:
                    payload = json.loads(message.content) if isinstance(message.content, str) else message.content
                except Exception:
                    payload = {}
                if not isinstance(payload, dict):
                    payload = {}
                prior_args = prior_bridge[2] if prior_bridge else {}
                query = extract_bridge_query(prior_args, spec.query_arg_candidates) or extract_bridge_query(
                    payload, spec.query_arg_candidates
                )
                display_name = prior_bridge[1] if prior_bridge else message.name
                bridge_requests.append(
                    BridgeExecution(
                        requested_tool_name=display_name,
                        display_tool_name=display_name,
                        builtin_tool_type=spec.builtin_tool_type,
                        args={"query": query} if query else {},
                    )
                )
                if spec.builtin_tool_type == "web_search":
                    if builtin_cfg is None:
                        from app.schemas.compat import BuiltinToolsConfig

                        builtin_cfg = BuiltinToolsConfig(web_search=True)
                    else:
                        builtin_cfg.web_search = True
                    if query:
                        bridge_messages.append(
                            {"role": "user", "content": f"Use web search to answer this query: {query}"}
                        )
                elif spec.builtin_tool_type == "file_search":
                    if builtin_cfg is None:
                        from app.schemas.compat import BuiltinToolsConfig

                        builtin_cfg = BuiltinToolsConfig(file_search={})
                    else:
                        builtin_cfg.file_search = builtin_cfg.file_search or {}
                    if query:
                        bridge_messages.append(
                            {"role": "user", "content": f"Use file search to answer this query: {query}"}
                        )
                continue

            forwarded_messages.append(message.model_dump())

        # Remove bridged tools from custom tool list so the model does not emit a custom function_call for them.
        filtered_custom_tools = []
        for t in custom_tools:
            if find_bridge_for_tool_name(t.function.name):
                continue
            filtered_custom_tools.append(t)

        input_payload = list(req.x_openai.input_items) if req.x_openai else []
        input_payload.extend(to_responses_input(forwarded_messages))
        input_payload.extend(bridge_messages)
        custom_responses_tools = to_responses_custom_tools(filtered_custom_tools)
        merged_tools = merge_builtin_tools(custom_responses_tools, builtin_cfg)
        include = build_include_list(
            builtin_cfg,
            include_web_search_results=self.include_web_search_results,
        )

        return model, input_payload, merged_tools, include, bridge_requests

    def run_nonstream(self, req):
        model, input_payload, tools, include, bridge_requests = self._prepare_request(req)
        initial_tool_choice = self._normalize_tool_choice(req.tool_choice)
        resp = self.adapter.create_response(
            model=model,
            input_payload=input_payload,
            tools=tools,
            tool_choice=initial_tool_choice,
            temperature=req.temperature,
            top_p=req.top_p,
            reasoning={"effort": req.reasoning_effort} if req.reasoning_effort is not None else None,
            max_output_tokens=req.max_completion_tokens or req.max_tokens,
            text=to_responses_text_config(req.response_format, req.verbosity),
            parallel_tool_calls=req.parallel_tool_calls,
            service_tier=req.service_tier,
            include=include,
            stream=False,
        )

        normalized = normalize_final_response(resp, bridge_executions=bridge_requests)
        self.audit_logger.log_chat(req, normalized)
        return normalized

    def run_stream(self, req):
        model, input_payload, tools, include, bridge_requests = self._prepare_request(req)
        openai_stream = self.adapter.create_response(
            model=model,
            input_payload=input_payload,
            tools=tools,
            tool_choice=self._normalize_tool_choice(req.tool_choice),
            temperature=req.temperature,
            top_p=req.top_p,
            reasoning={"effort": req.reasoning_effort} if req.reasoning_effort is not None else None,
            max_output_tokens=req.max_completion_tokens or req.max_tokens,
            text=to_responses_text_config(req.response_format, req.verbosity),
            parallel_tool_calls=req.parallel_tool_calls,
            service_tier=req.service_tier,
            include=include,
            stream=True,
        )
        return map_stream_events(
            openai_stream,
            model,
            bridge_executions=bridge_requests,
            include_usage=bool(req.stream_options and req.stream_options.get("include_usage")),
        )
