"""
Export trained PyTorch model to ONNX format.

Supports:
- FP32/FP16 export
- Opset version selection
- Model verification
"""

import argparse
from pathlib import Path

import torch
import onnx

from model import create_model_main, create_model_mini


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    config: str = "main",
    fp16: bool = False,
    opset_version: int = 14,
):
    """
    Export PyTorch model to ONNX.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Output ONNX file path
        config: Model config (main/mini)
        fp16: Export in FP16 (reduces size)
        opset_version: ONNX opset version
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    # Load model
    if config == "main":
        model = create_model_main()
    else:
        model = create_model_mini()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Convert to FP16 if requested
    if fp16:
        print("Converting to FP16...")
        model = model.half()

    # Create dummy input
    dummy_input = torch.randn(1, 18, 8, 8)
    if fp16:
        dummy_input = dummy_input.half()

    print(f"Exporting to ONNX (opset {opset_version})...")

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
    )

    # Verify
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Get file size
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\n✓ Export successful!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Format: {'FP16' if fp16 else 'FP32'}")


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--ckpt", type=str, required=True, help="PyTorch checkpoint")
    parser.add_argument("--out", type=str, required=True, help="Output ONNX file")
    parser.add_argument("--config", type=str, choices=["main", "mini"], default="main")
    parser.add_argument("--fp16", action="store_true", help="Export in FP16")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.out).parent
    output_dir.mkdir(exist_ok=True, parents=True)

    export_to_onnx(
        checkpoint_path=args.ckpt,
        output_path=args.out,
        config=args.config,
        fp16=args.fp16,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
