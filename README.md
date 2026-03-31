# OpenAI-compatible FastAPI gateway for Responses API

This project exposes a `POST /v1/chat/completions` endpoint that looks like Chat Completions to existing apps, while using OpenAI's Responses API internally.

Implemented scope:
- non-stream chat completions
- streaming SSE in chat.completion.chunk style
- custom function tools in non-stream mode
- bridge selected custom tools to OpenAI built-in tools
- built-in `web_search`, `file_search`, `code_interpreter`
- legacy display/log reconstruction (`assistant -> tool -> assistant` style)
- stateless gateway design
- `store=False` enforced on all Responses API calls

## Environment

- `OPENAI_API_KEY` - required
- `OPENAI_MODEL_DEFAULT` - optional, default model alias fallback

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

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

## Custom-tool bridge

If the request includes function tools such as `web_search`, `search_web`, or `browser_search`, the gateway can intercept those names and satisfy them with OpenAI built-in `web_search`, while emitting legacy-looking tool messages for UI/log compatibility.

Bridge mapping lives in `app/tools/registry.py`.
