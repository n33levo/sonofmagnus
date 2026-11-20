import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from chessio.lc0 import encode_history
from chessio.records import record_board
from model.artifacts import artifact_info
from play.mlx import Graph
from train.dataset import ChessPositionDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.max <= 0 or args.batch_size <= 0:
        parser.error('Positive sample and batch counts required')
    graph = Graph(args.model)
    if graph.quantized:
        parser.error('Calibrate the unquantized source')
    dataset = ChessPositionDataset(args.data, augment=False)
    if not len(dataset):
        parser.error('Empty calibration data')
    collect = mx.compile(lambda inputs, weights: graph.forward(inputs, weights, collect=True)[1])
    indices = np.random.default_rng(args.seed).choice(len(dataset), size=min(args.max, len(dataset)), replace=False)
    moments = {}
    for start in range(0, len(indices), args.batch_size):
        inputs = mx.array(np.stack([encode_history(record_board(dataset.read_row(int(i))))
            for i in indices[start:start + args.batch_size]]).astype(np.float16))
        batch = collect(inputs, graph.weights)
        moments = {name: value + moments.get(name, 0) for name, value in batch.items()}
        mx.eval(moments)
        print(f'Calibrated {min(start + args.batch_size, len(indices))}/{len(indices)}', flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out), moments)
    out.with_suffix('.json').write_text(json.dumps({'model': artifact_info(args.model), 'data': artifact_info(args.data),
        'samples': len(indices), 'seed': args.seed, 'matrices': len(moments), 'artifact': artifact_info(out)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
