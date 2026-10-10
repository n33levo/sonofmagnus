import argparse
import json
import time
from pathlib import Path

import chess
import numpy as np
import onnxruntime as ort

from chessio.encode import encode_board


def check_model(model_path, size_limit_mb=10., latency_target_ms=100., num_warmup=10, num_runs=100):
    if num_runs <= 0 or num_warmup < 0:
        raise ValueError('Runs must be positive and warmup nonnegative')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_path), options, providers=['CPUExecutionProvider'])
    input_type = session.get_inputs()[0].type
    sample = encode_board(chess.Board())[None].astype(np.float16 if input_type == 'tensor(float16)' else np.float32)
    name = session.get_inputs()[0].name
    for _ in range(num_warmup):
        session.run(None, {name: sample})
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        outputs = session.run(None, {name: sample})
        latencies.append((time.perf_counter() - start) * 1000)
        if not all(np.isfinite(value).all() for value in outputs):
            raise FloatingPointError('Non-finite ONNX output')
    size = Path(model_path).stat().st_size / 1e6
    results = {'file_size_mb': size, 'size_ok': size <= size_limit_mb,
               'mean_latency_ms': float(np.mean(latencies)),
               **{f'p{q}_latency_ms': float(np.percentile(latencies, q)) for q in (50, 95, 99)},
               'latency_ok': bool(np.percentile(latencies, 95) <= latency_target_ms),
               'provider': 'CPUExecutionProvider', 'runs': num_runs, 'threads': 4}
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--size-limit', type=float, default=10.)
    parser.add_argument('--latency-target', type=float, default=100.)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--out')
    args = parser.parse_args()
    results = check_model(args.model, args.size_limit, args.latency_target, args.warmup, args.runs)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + '\n')
    if not (results['size_ok'] and results['latency_ok']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
