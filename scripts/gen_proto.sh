#!/usr/bin/env bash
# 生成 Python protobuf 绑定
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python/.venv/bin/python
$PY -m grpc_tools.protoc \
  -I proto \
  --python_out=python/avatar/protocol \
  --pyi_out=python/avatar/protocol \
  proto/drive.proto
touch python/avatar/protocol/__init__.py
echo "generated:"
ls python/avatar/protocol/
