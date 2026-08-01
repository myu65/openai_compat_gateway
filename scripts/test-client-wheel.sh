#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_version="${1:-3.13}"
test_root="$(mktemp -d)"
trap 'rm -rf "${test_root}"' EXIT

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/openai-compat-gateway-uv-cache}"

uv build \
  --project "${repo_root}/packages/client" \
  --wheel \
  --out-dir "${test_root}/dist"

wheel="$(find "${test_root}/dist" -maxdepth 1 -name '*.whl' -print -quit)"
test -n "${wheel}"

WHEEL="${wheel}" python - <<'PY'
import os
from email.parser import BytesParser
from zipfile import ZipFile

wheel = os.environ["WHEEL"]
with ZipFile(wheel) as archive:
    names = archive.namelist()
    assert not any(name.startswith("app/") for name in names), names
    metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
    metadata = BytesParser().parsebytes(archive.read(metadata_name))

requirements = metadata.get_all("Requires-Dist", [])
assert requirements == ["langchain-openai<=1.4.1,>=0.3.35"], requirements
assert not any("fastapi" in requirement.lower() or "uvicorn" in requirement.lower() for requirement in requirements)
PY

uv venv "${test_root}/venv" --python "${python_version}" --seed
uv pip install --python "${test_root}/venv/bin/python" "${wheel}"
"${test_root}/venv/bin/python" -c \
  'from openai_compat_gateway_client import ChatOpenAICompat; print(ChatOpenAICompat.__name__)'
