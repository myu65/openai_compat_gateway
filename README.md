# OpenAI Chat Completions compatibility gateway

An OpenAI-compatible `POST /v1/chat/completions` endpoint with two execution paths:

- native Chat Completions pass-through when no translation is needed;
- Responses API translation when reasoning + tools, built-in tools, or lossless Responses state are required.

Every upstream request sets `store=false`. The gateway does not keep conversation state. It returns the Responses output items to the client so encrypted reasoning, built-in tool activity, and function calls can be stored in the application's own database and replayed on the next turn.

## Why two paths

Chat Completions and Responses do not have identical data models. A message-only conversion loses Responses reasoning and built-in-tool items, while forcing every request through Responses loses Chat Completions-only behavior such as `n` and `stop`.

| Request | `auto` route |
| --- | --- |
| Ordinary Chat Completions request | Native Chat Completions |
| `reasoning_effort` together with custom tools | Responses |
| `x_builtin_tools` | Responses |
| Bridged `web_search` / `file_search` function name | Responses |
| Replayed `x_openai` state | Responses |

Set `x_openai.mode` to `responses` or `chat_completions` to override routing. If a Responses request contains a feature that cannot be translated without changing meaning, the gateway returns an OpenAI-shaped 400 error instead of silently dropping it.

## Supported behavior

- custom function calls, including parallel calls and streamed arguments;
- client-side LangChain / Chat Completions tool loops;
- `reasoning_effort`, `temperature`, `top_p`, output-token limits, structured output, verbosity, `parallel_tool_calls`, and `service_tier` mapping;
- stateless reasoning continuity through encrypted reasoning items;
- native and translated SSE with real SSE delimiters and optional usage chunk;
- built-in web search, file search, and Code Interpreter;
- citations, file-search results, Code Interpreter outputs, and legacy display steps under `x_openai`;
- OpenAI-shaped validation, authentication, and upstream API errors;
- optional bearer authentication for the gateway itself.

Code Interpreter automatically uses an OpenAI auto container unless a container configuration is supplied.

## Install and run

```bash
cp .env.sample .env
uv sync --all-extras --dev
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Required environment variable:

- `OPENAI_API_KEY`: upstream OpenAI API key. Keep it on the gateway; do not send it from browser clients.

Optional environment variables:

- `OPENAI_MODEL_DEFAULT`: defaults to `gpt-5.6-luna`;
- `GATEWAY_API_KEY`: when set, clients must send this value as `Authorization: Bearer ...`;
- `OPENAI_INCLUDE_WEB_SEARCH_RESULTS`: enables `web_search_call.results` only for projects that permit that include value.

Container build:

```bash
docker build -t openai-compat-gateway .
docker run --rm -p 8000:8000 --env-file .env openai-compat-gateway
```

## Basic request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer YOUR_GATEWAY_API_KEY' \
  -d '{
    "model": "gpt-5.6-terra",
    "messages": [{"role": "user", "content": "京都の明日の天気を調べて"}],
    "reasoning_effort": "medium",
    "x_builtin_tools": {"web_search": true}
  }'
```

Omit the authorization header when `GATEWAY_API_KEY` is not configured.

## Lossless multi-turn state

In Responses mode, the assistant message and the response root include an extension like this:

```json
{
  "x_openai": {
    "mode": "responses",
    "response_items": [
      {
        "type": "reasoning",
        "encrypted_content": "..."
      },
      {
        "type": "message",
        "role": "assistant",
        "content": []
      }
    ]
  }
}
```

Persist the returned assistant message unchanged. When it is included in the next request, the gateway replays `message.x_openai.response_items` at that exact transcript position. This is the lossless path for `store=false`; do not combine it with `previous_response_id`.

Clients that store state separately can instead send prior items as a prefix:

```json
{
  "x_openai": {
    "mode": "responses",
    "input_items": [
      {"type": "reasoning", "encrypted_content": "..."}
    ]
  },
  "messages": [{"role": "user", "content": "continue"}]
}
```

For streaming, capture `x_openai` from the final chunk and attach it to the persisted assistant message. The final chunk also puts the same state in `delta.x_openai` for extension-aware clients.

## LangChain

Stock `langchain-openai` discards unknown Chat Completions fields while converting a response into `AIMessage`. Use the included subclass when reasoning or built-in-tool continuity matters:

```python
from app.integrations.langchain import ChatOpenAICompat

llm = ChatOpenAICompat(
    model="gpt-5.6-terra",
    base_url="http://localhost:8000/v1",
    api_key="YOUR_GATEWAY_API_KEY",
    reasoning_effort="medium",
)
```

`ChatOpenAICompat` stores the gateway state in `AIMessage.additional_kwargs["x_openai"]` and sends it back with that assistant message on later turns. Custom tools remain normal LangChain tools: the gateway returns `message.tool_calls`, LangChain executes them, and the next request supplies `role="tool"` output.

## Built-in tool bridge

Built-ins can be enabled explicitly:

```json
{
  "x_builtin_tools": {
    "web_search": true,
    "file_search": {"vector_store_ids": ["vs_123"]},
    "code_interpreter": {"container": {"type": "auto", "memory_limit": "4g"}}
  }
}
```

For older clients, function names `web_search`, `search_web`, `browser_search`, and `file_search` are recognized and converted to Responses built-ins. Built-ins execute at OpenAI; they are observational events, not fake client-side function calls. Legacy-looking display steps are available at `x_openai.legacy_steps`.

## Tests

```bash
./scripts/test-fast.sh
```

Real OpenAI tests are opt-in and never read a key from command-line arguments:

```bash
OPENAI_API_KEY=... RUN_OPENAI_E2E=1 ./scripts/test-e2e.sh
```

The suite covers request mapping, encrypted reasoning replay, custom tool round-trips, parallel streaming calls, real SSE framing, built-in tools, native/Responses routing, and LangChain state preservation.

## Data handling

- The gateway always sends `store=false` upstream.
- `previous_response_id` and the Conversations API are not used.
- Encrypted reasoning is opaque and is only round-tripped.
- Application logs should be treated as sensitive because prompts and tool results may contain business data.

The state design follows OpenAI's guidance to preserve and replay every returned reasoning item for stateless Responses usage: [Migrate to the Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses). Built-in configuration follows the current [Using tools](https://developers.openai.com/api/docs/guides/tools) and [Code Interpreter](https://developers.openai.com/api/docs/guides/tools-code-interpreter) guides.
