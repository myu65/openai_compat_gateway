from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from app.mappers.request_mapper import build_include_list, merge_builtin_tools, to_responses_custom_tools, to_responses_input
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
    ):
        self.adapter = adapter
        self.tool_executor = tool_executor
        self.audit_logger = audit_logger
        self.default_model = default_model
        self.include_web_search_results = include_web_search_results

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
        model = req.model or self.default_model
        builtin_cfg = deepcopy(req.x_builtin_tools)
        custom_tools = list(req.tools or [])
        bridge_requests: list[BridgeExecution] = []
        bridge_messages: list[dict[str, Any]] = []

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

        # Bridge selected custom tools to built-in tools based on prior tool results already in messages.
        for message in req.messages:
            if message.role == "tool" and message.name:
                spec = find_bridge_for_tool_name(message.name)
                if not spec:
                    continue
                try:
                    payload = json.loads(message.content) if isinstance(message.content, str) else message.content
                except Exception:
                    payload = {}
                if not isinstance(payload, dict):
                    payload = {}
                query = extract_bridge_query(payload, spec.query_arg_candidates)
                bridge_requests.append(
                    BridgeExecution(
                        requested_tool_name=message.name,
                        display_tool_name=message.name,
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
                        bridge_messages.append({"role": "user", "content": f"Use web search to answer this query: {query}"})
                elif spec.builtin_tool_type == "file_search":
                    if builtin_cfg is None:
                        from app.schemas.compat import BuiltinToolsConfig
                        builtin_cfg = BuiltinToolsConfig(file_search={})
                    else:
                        builtin_cfg.file_search = builtin_cfg.file_search or {}
                    if query:
                        bridge_messages.append({"role": "user", "content": f"Use file search to answer this query: {query}"})

        # Remove bridged tools from custom tool list so the model does not emit a custom function_call for them.
        filtered_custom_tools = []
        for t in custom_tools:
            if find_bridge_for_tool_name(t.function.name):
                continue
            filtered_custom_tools.append(t)

        input_payload = to_responses_input([m.model_dump() for m in req.messages])
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
        rolling_input = list(input_payload)
        initial_tool_choice = self._normalize_tool_choice(req.tool_choice)
        resp = self.adapter.create_response(
            model=model,
            input_payload=rolling_input,
            tools=tools,
            tool_choice=initial_tool_choice,
            temperature=req.temperature,
            include=include,
            stream=False,
        )

        while True:
            function_calls = [i for i in getattr(resp, "output", []) or [] if getattr(i, "type", None) == "function_call"]
            function_calls = [fc for fc in function_calls if self.tool_executor.has(fc.name)]
            if not function_calls:
                break

            followup_input: list[dict[str, Any]] = []
            for fc in function_calls:
                followup_input.append(
                    {
                        "type": "function_call",
                        "call_id": fc.call_id,
                        "name": fc.name,
                        "arguments": fc.arguments,
                    }
                )
                result = self.tool_executor.execute(fc.name, fc.arguments)
                followup_input.append({"type": "function_call_output", "call_id": fc.call_id, "output": result})

            rolling_input.extend(followup_input)
            resp = self.adapter.create_response(
                model=model,
                input_payload=rolling_input,
                tools=tools,
                tool_choice=self._tool_choice_for_followup(req.tool_choice),
                temperature=req.temperature,
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
            include=include,
            stream=True,
        )
        return map_stream_events(openai_stream, model, bridge_executions=bridge_requests)
