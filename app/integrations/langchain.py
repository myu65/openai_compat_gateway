"""Backward-compatible source-checkout import.

Applications should depend on ``packages/client`` and import from
``openai_compat_gateway_client`` directly.
"""

from openai_compat_gateway_client import ChatOpenAICompat

__all__ = ["ChatOpenAICompat"]
