#!/usr/bin/env bash
set -euo pipefail

PY="/opt/venv/bin/python"

echo "[postCreate] Using python: $($PY -c 'import sys; print(sys.executable)')"
$PY -m pip install -U pip
$PY -m pip install -e ".[dev]"

echo "[postCreate] Done."
