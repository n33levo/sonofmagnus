"""
Size and latency verification for deployment.

Ensures model meets:
- Size requirement (≤10 MB for Knight's Edge)
- Latency targets
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def check_model(
    model_path: str,
    size_limit_mb: float = 10.0,
    latency_target_ms: float = 100.0,
    num_warmup: int = 10,
    num_runs: int = 100,
):
    """
    Check model size and latency.

    Args:
        model_path: Path to ONNX model
        size_limit_mb: Size limit in MB
        latency_target_ms: Latency target in ms
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs

    Returns:
        Dictionary with results
    """
    print(f"Checking model: {model_path}")

    # Check file size
    file_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"\nFile size: {file_size_mb:.2f} MB")

    size_ok = file_size_mb <= size_limit_mb
    if size_ok:
        print(f"✓ Size check passed (≤{size_limit_mb} MB)")
    else:
        print(f"✗ Size check FAILED (>{size_limit_mb} MB)")

    # Load model
    print(f"\nLoading model with ONNX Runtime...")
    session = ort.InferenceSession(model_path)

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]

    # Create dummy input
    dummy_input = np.random.randn(1, 18, 8, 8).astype(np.float32)

    # Warmup
    print(f"Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        _ = session.run(output_names, {input_name: dummy_input})

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    latencies = []

    for _ in range(num_runs):
        start = time.time()
        _ = session.run(output_names, {input_name: dummy_input})
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

    # Statistics
    mean_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    print(f"\nLatency statistics:")
    print(f"  Mean: {mean_latency:.2f} ms")
    print(f"  P50:  {p50_latency:.2f} ms")
    print(f"  P95:  {p95_latency:.2f} ms")
    print(f"  P99:  {p99_latency:.2f} ms")

    latency_ok = mean_latency <= latency_target_ms
    if latency_ok:
        print(f"✓ Latency check passed (≤{latency_target_ms} ms)")
    else:
        print(f"✗ Latency check FAILED (>{latency_target_ms} ms)")

    # Overall result
    print("\n" + "="*50)
    if size_ok and latency_ok:
        print("✓ ALL CHECKS PASSED")
    else:
        print("✗ SOME CHECKS FAILED")
    print("="*50)

    return {
        "file_size_mb": file_size_mb,
        "size_ok": size_ok,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
        "latency_ok": latency_ok,
    }


def main():
    parser = argparse.ArgumentParser(description="Check model size and latency")
    parser.add_argument("--model", type=str, required=True, help="ONNX model path")
    parser.add_argument("--size-limit", type=float, default=10.0, help="Size limit in MB")
    parser.add_argument("--latency-target", type=float, default=100.0, help="Latency target in ms")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=100, help="Benchmark runs")
    args = parser.parse_args()

    results = check_model(
        model_path=args.model,
        size_limit_mb=args.size_limit,
        latency_target_ms=args.latency_target,
        num_warmup=args.warmup,
        num_runs=args.runs,
    )

    # Exit with error if checks failed
    if not (results["size_ok"] and results["latency_ok"]):
        exit(1)


if __name__ == "__main__":
    main()
