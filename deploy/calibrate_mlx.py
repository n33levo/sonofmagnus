import argparse
import json
from pathlib import Path

import chess
import mlx.core as mx
import numpy as np

from chessio.lc0 import encode_history
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
    args = parser.parse_args()
    if args.max <= 0 or args.batch_size <= 0:
        parser.error('Positive sample and batch counts required')
    graph = Graph(args.model)
    if graph.quantized:
        parser.error('Calibrate the unquantized source')
    dataset = ChessPositionDataset(args.data, augment=False, max_samples=args.max)
    if not len(dataset):
        parser.error('Empty calibration data')
    collect = mx.compile(lambda inputs, weights: graph.forward(inputs, weights, collect=True)[1])
    moments = {}
    for start in range(0, len(dataset), args.batch_size):
        inputs = mx.array(np.stack([encode_history(chess.Board(dataset.read_row(i)['fen']))
            for i in range(start, min(start + args.batch_size, len(dataset)))]).astype(np.float16))
        batch = collect(inputs, graph.weights)
        moments = {name: value + moments.get(name, 0) for name, value in batch.items()}
        mx.eval(moments)
        print(f'Calibrated {min(start + args.batch_size, len(dataset))}/{len(dataset)}', flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out), moments)
    out.with_suffix('.json').write_text(json.dumps({'model': artifact_info(args.model), 'data': artifact_info(args.data),
        'samples': len(dataset), 'matrices': len(moments), 'artifact': artifact_info(out)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
