from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_env_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return parsed


def _parse_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be 0 or greater")
    return parsed


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        os.environ.setdefault(key, _parse_env_value(value))


_load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model_default: str = os.getenv("OPENAI_MODEL_DEFAULT", "gpt-5.6-luna")
    openai_include_web_search_results: bool = _parse_env_bool("OPENAI_INCLUDE_WEB_SEARCH_RESULTS", default=False)
    openai_connect_timeout_seconds: float = _parse_env_float("OPENAI_CONNECT_TIMEOUT_SECONDS", 10.0)
    openai_read_timeout_seconds: float = _parse_env_float("OPENAI_READ_TIMEOUT_SECONDS", 900.0)
    openai_write_timeout_seconds: float = _parse_env_float("OPENAI_WRITE_TIMEOUT_SECONDS", 30.0)
    openai_pool_timeout_seconds: float = _parse_env_float("OPENAI_POOL_TIMEOUT_SECONDS", 10.0)
    openai_max_retries: int = _parse_env_int("OPENAI_MAX_RETRIES", 0)
    gateway_api_key: str | None = os.getenv("GATEWAY_API_KEY")


settings = Settings()
