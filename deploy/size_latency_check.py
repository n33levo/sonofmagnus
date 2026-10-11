import argparse
import json
from pathlib import Path

import chess
import onnx

from eval.latency import benchmark, sample_positions
from model.artifacts import artifact_info
from play.onnx import OnnxRunner


def check_model(model_path, size_limit_mb=8.7, latency_target_ms=100., num_warmup=10, num_runs=100, pgn_path=None):
    model = onnx.load(model_path, load_external_data=False)
    if any(weight.external_data for weight in model.graph.initializer):
        raise ValueError('Export a self-contained ONNX artifact before measuring its size')
    positions = sample_positions(pgn_path, num_runs) if pgn_path else [chess.Board()]
    timing = benchmark(OnnxRunner(model_path), positions, num_runs, num_warmup)
    artifact = artifact_info(model_path)
    size = artifact['model_bytes'] / 1e6
    results = {'file_size_mb': size, 'size_ok': size <= size_limit_mb,
               **{f'p{q}_latency_ms': timing[f'p{q}_ms'] for q in (50, 95, 99)},
               'latency_ok': timing['p95_ms'] < latency_target_ms,
               'provider': 'CPUExecutionProvider', 'runs': num_runs, 'threads': 4,
               'positions': len(positions), 'neural_forward_calls': timing['neural_forward_calls'],
               'protocol': 'Complete move selection; retained PGN history' if pgn_path else 'Complete move selection; starting position only',
               'model_sha256': artifact['sha256']}
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--size-limit', type=float, default=8.7)
    parser.add_argument('--latency-target', type=float, default=100.)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--pgn', help='Retain real game histories in the latency benchmark')
    parser.add_argument('--out')
    args = parser.parse_args()
    results = check_model(args.model, args.size_limit, args.latency_target, args.warmup, args.runs, args.pgn)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + '\n')
    if not (results['size_ok'] and results['latency_ok']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
