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
- `OPENAI_INCLUDE_WEB_SEARCH_RESULTS`: enables `web_search_call.results` only for projects that permit that include value;
- `OPENAI_CONNECT_TIMEOUT_SECONDS`: upstream connection timeout, default `10` seconds;
- `OPENAI_READ_TIMEOUT_SECONDS`: upstream response read timeout, default `900` seconds. This is intentionally long because LLM generation and reasoning can take several minutes;
- `OPENAI_WRITE_TIMEOUT_SECONDS`: upstream request write timeout, default `30` seconds;
- `OPENAI_POOL_TIMEOUT_SECONDS`: wait-for-connection-pool timeout, default `10` seconds;
- `OPENAI_MAX_RETRIES`: OpenAI SDK retry count, default `0` so a request that already consumed the long LLM read timeout is not automatically repeated for another full timeout window;
- `WEB_CONCURRENCY`: Uvicorn worker count for the production container, default `2`. Increase this when one slow upstream request should not consume too much of the gateway's total request capacity.

The timeout policy deliberately separates short infrastructure waits from long model generation. Connection and pool waits fail quickly, while response reads allow up to 15 minutes by default. An upstream OpenAI timeout is returned by the gateway as HTTP `504` instead of leaving the request hanging indefinitely.

For local development, the command above runs one Uvicorn process. The production container defaults to two workers and can be overridden without rebuilding the image, for example:

```bash
WEB_CONCURRENCY=4 docker run --rm -p 8000:8000 --env-file .env openai-compat-gateway
```

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

Stock `langchain-openai` discards unknown Chat Completions fields while converting a response into `AIMessage`. The standalone client package preserves those fields without installing this server, FastAPI, or uvicorn.

PyPI publication is not required. Install the package directly from this monorepo (its current default branch is `main`):

```bash
uv add "openai-compat-gateway-client @ git+https://github.com/myu65/openai_compat_gateway.git@main#subdirectory=packages/client"
```

```bash
pip install "openai-compat-gateway-client @ git+https://github.com/myu65/openai_compat_gateway.git@main#subdirectory=packages/client"
```

Using the default branch is convenient during development. For production, replace `main` with a release tag or a full commit SHA so dependency resolution is reproducible.

The client supports `langchain-openai>=0.3.35,<=1.4.1`. CI covers `0.3.35`, each `1.x` minor through `1.4.1`, and both Python 3.11 and 3.13 at the supported boundaries. The upper bound is intentional because the client overrides private `ChatOpenAI` methods; raise it only together with the private-API contract and round-trip tests.

### Configuration and invoke

```python
from langchain_core.messages import HumanMessage
from openai_compat_gateway_client import ChatOpenAICompat

llm = ChatOpenAICompat(
    model="gpt-5.6-terra",
    base_url="http://localhost:8000/v1",
    api_key="YOUR_GATEWAY_API_KEY",
    reasoning_effort="medium",
)

answer = llm.invoke([HumanMessage(content="Explain the result briefly.")])
print(answer.content)
```

The client's `api_key` is `GATEWAY_API_KEY`, not the upstream OpenAI API key. `OPENAI_API_KEY` belongs only on the gateway server and must not be shipped to or persisted by the application.

`ChatOpenAICompat` stores encrypted reasoning and other Responses items in `AIMessage.additional_kwargs["x_openai"]`, then restores that state on the corresponding assistant message in later requests.

### Two-turn custom tool loop

Custom tools remain normal LangChain tools. `bind_tools` exposes their schemas; the application executes each returned call and appends a `ToolMessage` before asking for the final answer:

```python
import json

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


@tool
def lookup_order(order_id: str) -> dict[str, str]:
    """Look up an order by id."""
    return {"order_id": order_id, "status": "shipped", "tracking_code": "ZX-31415"}


tool_llm = llm.bind_tools([lookup_order])
messages = [HumanMessage(content="Check order ord_314, then report its status.")]

assistant = tool_llm.invoke(messages)
messages.append(assistant)  # includes tool_calls and additional_kwargs["x_openai"]

for call in assistant.tool_calls:
    result = lookup_order.invoke(call["args"])
    messages.append(
        ToolMessage(
            content=json.dumps(result),
            tool_call_id=call["id"],
        )
    )

final = tool_llm.invoke(messages)
print(final.content)
```

This exact assistant message must remain before its matching `ToolMessage`. Its `x_openai.response_items` contains the reasoning item and upstream function-call state required for the second turn.

### Built-in tools

OpenAI built-ins execute inside the gateway's Responses path. Pass their configuration through `extra_body`:

```python
web_llm = ChatOpenAICompat(
    model="gpt-5.6-terra",
    base_url="http://localhost:8000/v1",
    api_key="YOUR_GATEWAY_API_KEY",
    extra_body={"x_builtin_tools": {"web_search": True}},
)

answer = web_llm.invoke("Find the latest official release notes.")
```

### Persisting messages

Do not reduce an `AIMessage` to only `role` and `content`. A database representation must also preserve `additional_kwargs` (including `x_openai`) and `tool_calls`. LangChain's serializer keeps the complete message shape:

```python
import json

from langchain_core.messages import message_to_dict, messages_from_dict

stored_json = json.dumps(message_to_dict(assistant))
restored_assistant = messages_from_dict([json.loads(stored_json)])[0]
```

If an application uses its own columns, store at least `role`/message type, `content`, `additional_kwargs`, and `tool_calls`, then reconstruct the full `AIMessage`. LangGraph checkpointers that persist the complete LangChain message retain the state automatically.

### Streaming

The gateway emits `x_openai` on the final stream chunk. Aggregate every chunk before converting it to the assistant message that is stored and replayed:

```python
from langchain_core.messages import HumanMessage, message_chunk_to_message

chunks = iter(llm.stream([HumanMessage(content="Think through this problem.")]))
combined = next(chunks)
for chunk in chunks:
    print(chunk.content, end="", flush=True)
    combined += chunk

assistant = message_chunk_to_message(combined)
assert "x_openai" in assistant.additional_kwargs
```

Persist `assistant` exactly as described above. Supplying it in the next transcript resends the streaming final state at the correct assistant position.

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

Build the client wheel, inspect its metadata and contents, install it into a clean environment, and verify its public import with:

```bash
./scripts/test-client-wheel.sh 3.11
./scripts/test-client-wheel.sh 3.13
```

Real OpenAI tests are opt-in and never read a key from command-line arguments:

```bash
OPENAI_API_KEY=... RUN_OPENAI_E2E=1 ./scripts/test-e2e.sh
```

The manual `OpenAI E2E` workflow uses the protected `openai-e2e` Environment. It exercises the local gateway HTTP endpoint against the real Responses API and covers:

- a Web Search result persisted through LangChain's JSON message serializer, replayed at the original assistant position, and then used to produce a custom LangChain tool call;
- a mixed transcript containing built-in state, a custom function call, and an application `ToolMessage`, with both results used in the final turn;
- streaming Code Interpreter output containing a hidden random value, followed by a restored second turn that computes the value's SHA-256;
- direct structural assertions on the second upstream Responses input, using only item types, IDs, and SHA-256 fingerprints so prompts, tool output, and encrypted reasoning are not logged.

For `store=false`, `x_openai.response_items` is the replay-critical field and must be kept on its original assistant message. Persist the complete `additional_kwargs["x_openai"]`, not only `response_items`: citations, sources, built-in events, Code Interpreter outputs, response status, and encrypted reasoning are useful for display, audit, and future-compatible replay. Custom-tool assistants must also retain `tool_calls`, and each matching `ToolMessage` must retain its `tool_call_id` and content.

The full suite also covers request mapping, parallel streaming calls, real SSE framing, native/Responses routing, wheel installation, and the supported `langchain-openai` compatibility range.

## Data handling

- The gateway always sends `store=false` upstream.
- `previous_response_id` and the Conversations API are not used.
- Encrypted reasoning is opaque and is only round-tripped.
- Application logs should be treated as sensitive because prompts and tool results may contain business data.

The state design follows OpenAI's guidance to preserve and replay every returned reasoning item for stateless Responses usage: [Migrate to the Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses). Built-in configuration follows the current [Using tools](https://developers.openai.com/api/docs/guides/tools) and [Code Interpreter](https://developers.openai.com/api/docs/guides/tools-code-interpreter) guides.
