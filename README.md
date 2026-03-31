# OpenAI-compatible FastAPI gateway for Responses API

This project exposes a `POST /v1/chat/completions` endpoint that looks like Chat Completions to existing apps, while using OpenAI's Responses API internally.

Implemented scope:
- non-stream chat completions
- streaming SSE in chat.completion.chunk style
- custom function tools in chat.completions-compatible mode
- bridge selected custom tools to OpenAI built-in tools
- built-in `web_search`, `file_search`, `code_interpreter`
- legacy display/log reconstruction (`assistant -> tool -> assistant` style)
- stateless gateway design
- `store=False` enforced on all Responses API calls

## Environment

- `OPENAI_API_KEY` - required
- `OPENAI_MODEL_DEFAULT` - optional, default model alias fallback
- `OPENAI_INCLUDE_WEB_SEARCH_RESULTS` - optional, defaults to `false`; enable only if your org has `include=web_search_call.results`
- `.env` is loaded automatically on startup if present

## Install

```bash
cp .env.sample .env
```

```bash
uv sync
```

## Run

```bash
uv run uvicorn app.main:app --reload --port 8000
```

## Test

```bash
./scripts/test-fast.sh
```

Run the real OpenAI E2E explicitly:

```bash
./scripts/test-e2e.sh
```

Available entry points:
- `./scripts/test-fast.sh` - default local/CI path, excludes real OpenAI E2E
- `./scripts/test-all.sh` - run the full pytest suite
- `./scripts/test-e2e.sh` - opt in to real OpenAI API tests with `RUN_OPENAI_E2E=1`
- `./scripts/run.sh` - start the FastAPI app locally

## Example request

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "gpt-5.4-mini",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "京都の明日の天気を調べて"}
    ],
    "stream": false,
    "x_builtin_tools": {"web_search": true}
  }' | jq
```

## Compatibility model

This gateway is intended to let existing `chat/completions` clients keep working while the backend uses the Responses API.

Tool behavior is split into two categories:
- custom function tools: OpenAI Chat Completions compatible pass-through
- built-in tools: gateway-side bridge to OpenAI Responses built-ins

### Custom function tools

Custom tools are treated as client-side tools, which matches the usual LangChain / Chat Completions flow:
- the client sends `tools` in Chat Completions format
- the gateway forwards them to the Responses API as function tools
- if the model decides to call a custom tool, the gateway returns `message.tool_calls`
- `finish_reason` is `tool_calls` for that turn
- the client executes the tool and sends a follow-up request with:
  - the prior assistant message containing `tool_calls`
  - one or more `role="tool"` messages with `tool_call_id`
- the gateway reconstructs Responses API `function_call` / `function_call_output` items from those messages

This means the gateway does not execute arbitrary custom tools on behalf of the client. Existing LangChain-style tool loops are expected to keep running on the client side.

### Built-in tool bridge

OpenAI built-in tools such as `web_search`, `file_search`, and `code_interpreter` do not behave like normal client-side custom tools. To keep older `chat/completions` clients usable, the gateway can bridge selected function-tool names to Responses built-ins.

Supported bridge names currently include:
- `web_search`
- `search_web`
- `browser_search`
- `file_search`

When one of those names is present in `tools`, the gateway:
- removes that tool from the custom function-tool list sent to the model
- enables the matching Responses built-in tool
- maps built-in execution back into legacy-looking `assistant -> tool -> assistant` logs in `x_openai.legacy_steps`

The bridge is intentionally opinionated. It exists to preserve compatibility for older clients that only know how to declare function tools, even when the underlying capability is really an OpenAI built-in.

### Finish reasons

The gateway returns Chat Completions-style finish reasons:
- `tool_calls` when the model turn ends with custom function calls
- `stop` when the model turn ends with a final assistant answer

Streaming responses follow the same rule in the final chunk.

### Follow-up request shape

For custom tools, the expected follow-up request looks like normal Chat Completions traffic:

```json
{
  "messages": [
    {"role": "user", "content": "u1を見て"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "lookup_profile",
            "arguments": "{\"user_id\":\"u1\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "content": "{\"role\":\"admin\"}"
    }
  ]
}
```

The gateway converts that follow-up into Responses API input items internally and then continues the turn against the Responses backend.

Bridge name mapping lives in `app/tools/registry.py`.
