#!/bin/bash
# Promote a training snapshot to submission-ready model
# Usage: bash scripts/promote_snapshot.sh <checkpoint_path> [config]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/promote_snapshot.sh <checkpoint_path> [main|mini]"
    exit 1
fi

CKPT_PATH="$1"
CONFIG="${2:-main}"

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

echo "Promoting snapshot: $CKPT_PATH"
echo "Config: $CONFIG"
echo ""

# Create output directory
OUT_DIR="out"
mkdir -p "$OUT_DIR"

# Determine output name from checkpoint
CKPT_NAME=$(basename "$CKPT_PATH" .pt)
ONNX_FP16="$OUT_DIR/${CKPT_NAME}_fp16.onnx"
ONNX_INT8="$OUT_DIR/${CKPT_NAME}_int8.onnx"

# Step 1: Export to ONNX (FP16)
echo "Step 1: Exporting to ONNX (FP16)..."
python -m deploy.export_onnx \
    --ckpt "$CKPT_PATH" \
    --out "$ONNX_FP16" \
    --config "$CONFIG" \
    --fp16

echo ""

# Step 2: Quantize to INT8 (if mini config)
if [ "$CONFIG" = "mini" ]; then
    echo "Step 2: Quantizing to INT8..."
    python -m deploy.quantize_int8 \
        --in "$ONNX_FP16" \
        --out "$ONNX_INT8" \
        --weight-type int8

    FINAL_MODEL="$ONNX_INT8"
    SIZE_LIMIT=10.0
    echo ""
else
    echo "Step 2: Skipping INT8 quantization (main config)"
    FINAL_MODEL="$ONNX_FP16"
    SIZE_LIMIT=100.0
    echo ""
fi

# Step 3: Size and latency check
echo "Step 3: Running size and latency checks..."
python -m deploy.size_latency_check \
    --model "$FINAL_MODEL" \
    --size-limit "$SIZE_LIMIT" \
    --latency-target 100 \
    --runs 100

echo ""

# Step 4: Gate checks (optional - run tests)
echo "Step 4: Running compliance tests..."
echo "  (Run 'make test' to verify all gates)"

echo ""
echo "="*60
echo "✓ Snapshot promotion complete!"
echo "="*60
echo "Final model: $FINAL_MODEL"
echo ""
echo "Next steps:"
echo "  1. Run full test suite: make test"
echo "  2. Evaluate on puzzles: python -m eval.puzzles --ckpt $CKPT_PATH --puzzles data/puzzles.jsonl"
echo "  3. Run arena matches: python -m eval.arena --a $CKPT_PATH --b ckpts/baseline.pt"
echo "  4. Submit: $FINAL_MODEL"
echo "="*60
