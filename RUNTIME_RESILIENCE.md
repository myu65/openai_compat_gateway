# Runtime resilience

The production gateway runtime uses `AsyncOpenAI` end to end so long-lived model requests and SSE streams do not consume Starlette's shared synchronous thread pool.

## Failure containment

- HTTP phase timeouts are explicit for connect, read, write, and connection-pool waits.
- `OPENAI_REQUEST_DEADLINE_SECONDS` is a wall-clock deadline. For streaming requests it spans upstream stream creation and later iteration, rather than resetting after the stream object is created.
- `OPENAI_MAX_CONNECTIONS` bounds upstream sockets per worker.
- `GATEWAY_MAX_INFLIGHT_REQUESTS` bounds long-lived requests per worker. Requests above the limit fail immediately with HTTP 503, `code=gateway_busy`, and `Retry-After: 1` instead of waiting behind stalled work.
- OpenAI SDK retries default to zero so a request that already consumed a long deadline is not silently repeated.

## Streaming cleanup

Upstream streams are explicitly closed from `finally` paths on normal completion, upstream errors, total-deadline expiry, and downstream cancellation/disconnect. A downstream disconnect therefore releases the upstream HTTP response and its in-flight admission slot instead of leaving an orphaned model request occupying gateway capacity.

The OpenAI Python SDK currently has an unresolved streaming connection-reuse bug, openai-python issue #3440, where a stream can stop at `[DONE]` before the HTTP/1.1 chunk terminator has been drained. Until that is fixed upstream, the gateway defaults to:

```env
OPENAI_MAX_KEEPALIVE_CONNECTIONS=0
GATEWAY_STREAM_CONNECTION_CLOSE=true
```

The first setting prevents a potentially damaged gateway-to-OpenAI connection from being reused. The second sends `Connection: close` on gateway SSE responses so downstream clients using affected SDK versions do not reuse that connection either. These are reliability workarounds, not protocol requirements, and can be revisited after the upstream SDK fix is released and deployed.

## Observability

Each request receives an `X-Request-Id`. Runtime logs record request start/end, execution mode, streaming outcome, elapsed time, and active in-flight count. `/healthz` reports process liveness plus `active_requests`, `max_inflight_requests`, and whether the shared upstream client has been initialized.

## Relevant environment variables

```env
OPENAI_CONNECT_TIMEOUT_SECONDS=10
OPENAI_READ_TIMEOUT_SECONDS=900
OPENAI_WRITE_TIMEOUT_SECONDS=30
OPENAI_POOL_TIMEOUT_SECONDS=10
OPENAI_REQUEST_DEADLINE_SECONDS=1200
OPENAI_MAX_CONNECTIONS=64
OPENAI_MAX_KEEPALIVE_CONNECTIONS=0
OPENAI_MAX_RETRIES=0
GATEWAY_MAX_INFLIGHT_REQUESTS=64
GATEWAY_STREAM_CONNECTION_CLOSE=true
```

`OPENAI_READ_TIMEOUT_SECONDS` is an inactivity timeout for an individual network read. It is intentionally separate from `OPENAI_REQUEST_DEADLINE_SECONDS`, which caps total wall time even if the peer continues to trickle data.

`WEB_CONCURRENCY` remains useful for process isolation, but it is no longer the primary defense against stalled upstream calls because the production request path itself is asynchronous and bounded.
