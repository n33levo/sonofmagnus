"""
INT8 quantization for ONNX models.

Reduces model size for Knight's Edge track (≤10 MB requirement).
"""

import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_model(
    input_path: str,
    output_path: str,
    weight_type: str = "int8",
):
    """
    Quantize ONNX model to INT8.

    Args:
        input_path: Input ONNX model
        output_path: Output quantized ONNX model
        weight_type: Weight quantization type (int8/uint8)
    """
    print(f"Loading model from {input_path}")

    # Get input size
    input_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
    print(f"Input size: {input_size_mb:.2f} MB")

    # Quantize
    print(f"Quantizing to {weight_type.upper()}...")

    quant_type = QuantType.QInt8 if weight_type == "int8" else QuantType.QUInt8

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=quant_type,
    )

    # Verify
    print("Verifying quantized model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Get output size
    output_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    compression_ratio = input_size_mb / output_size_mb if output_size_mb > 0 else 0

    print(f"\n✓ Quantization successful!")
    print(f"  Output: {output_path}")
    print(f"  Size: {output_size_mb:.2f} MB")
    print(f"  Compression: {compression_ratio:.2f}x")

    if output_size_mb > 10.0:
        print(f"\n⚠ WARNING: Model size {output_size_mb:.2f} MB exceeds 10 MB limit for Knight's Edge!")
    else:
        print(f"\n✓ Model meets ≤10 MB requirement for Knight's Edge")


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument("--in", dest="input", type=str, required=True, help="Input ONNX model")
    parser.add_argument("--out", type=str, required=True, help="Output quantized ONNX model")
    parser.add_argument("--weight-type", type=str, choices=["int8", "uint8"], default="int8")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.out).parent
    output_dir.mkdir(exist_ok=True, parents=True)

    quantize_model(
        input_path=args.input,
        output_path=args.out,
        weight_type=args.weight_type,
    )


if __name__ == "__main__":
    main()
