# openai-compat-gateway-client

The minimal LangChain client for
[openai-compat-gateway](https://github.com/myu65/openai_compat_gateway). It
preserves the gateway's `x_openai` state across non-streaming and streaming
turns without installing the FastAPI server.

```bash
pip install "openai-compat-gateway-client @ git+https://github.com/myu65/openai_compat_gateway.git@master#subdirectory=packages/client"
```

```python
from openai_compat_gateway_client import ChatOpenAICompat
```

See the repository README for complete invoke, tool-loop, built-in tool, state
persistence, and streaming examples.
