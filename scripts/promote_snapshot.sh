#!/bin/bash
set -euo pipefail

CKPT_PATH="${1:?Usage: promote_snapshot.sh checkpoint [main|mini] [calibration.jsonl]}"
CONFIG="${2:-main}"
PYTHON="${PYTHON:-.venv/bin/python}"
CKPT_NAME=$(basename "$CKPT_PATH" .pt)
mkdir -p out

"$PYTHON" -m pytest tests -q
if [ "$CONFIG" = mini ]; then
    CALIBRATION="${3:?Mini INT8 export requires a calibration JSONL}"
    "$PYTHON" -m deploy.export_onnx --ckpt "$CKPT_PATH" --out "out/${CKPT_NAME}_fp32.onnx" --config "$CONFIG"
    "$PYTHON" -m deploy.quantize_int8 --in "out/${CKPT_NAME}_fp32.onnx" --out "out/${CKPT_NAME}_int8.onnx" --calibration-data "$CALIBRATION"
    FINAL_MODEL="out/${CKPT_NAME}_int8.onnx"
    SIZE_LIMIT=8.7
else
    "$PYTHON" -m deploy.export_onnx --ckpt "$CKPT_PATH" --out "out/${CKPT_NAME}_fp16.onnx" --config "$CONFIG" --fp16
    FINAL_MODEL="out/${CKPT_NAME}_fp16.onnx"
    SIZE_LIMIT=24
fi
"$PYTHON" -m deploy.size_latency_check --model "$FINAL_MODEL" --size-limit "$SIZE_LIMIT" --latency-target 100
printf 'Verified deployment artifact: %s\n' "$FINAL_MODEL"
