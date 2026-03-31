#!/usr/bin/env bash
set -euo pipefail

RUN_OPENAI_E2E=1 uv run pytest -m e2e -s "$@"
