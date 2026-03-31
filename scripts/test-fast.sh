#!/usr/bin/env bash
set -euo pipefail

uv run pytest -m "not e2e" "$@"
