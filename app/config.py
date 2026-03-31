from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model_default: str = os.getenv("OPENAI_MODEL_DEFAULT", "gpt-5.4-mini")


settings = Settings()
