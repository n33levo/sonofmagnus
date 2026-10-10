import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from play.runner import load_runner
from model.artifacts import artifact_info


def sample_positions(pgn_path, count=200, seed=0):
    rng = random.Random(seed)
    sample, seen = [], 0
    with open(pgn_path) as stream:
        while (game := chess.pgn.read_game(stream)) is not None:
            if game.errors:
                raise ValueError('Malformed benchmark PGN')
            board = game.board()
            for move in game.mainline_moves():
                if not board.is_game_over(claim_draw=True):
                    seen += 1
                    index = len(sample) if len(sample) < count else rng.randrange(seen)
                    if index < count:
                        if index == len(sample):
                            sample.append(board.copy())
                        else:
                            sample[index] = board.copy()
                board.push(move)
    if not sample:
        raise ValueError('No playable benchmark positions')
    return sample


def benchmark(runner, boards, runs=200, warmup=20):
    if not boards or runs <= 0 or warmup < 0:
        raise ValueError('Positions and runs must be positive; warmup nonnegative')
    timings = []
    forward_start = getattr(runner, 'forward_calls', None)
    for idx in range(warmup + runs):
        board = boards[idx % len(boards)]
        start = time.perf_counter()
        move = runner.select_move(board, deterministic=True)
        elapsed = (time.perf_counter() - start) * 1000
        if move not in board.legal_moves:
            raise ValueError('Benchmark selected an illegal move')
        if idx >= warmup:
            timings.append(elapsed)
    result = {'runs': runs, 'warmup': warmup, 'positions': len(boards),
              'protocol': 'full move selection, deterministic, PGN history retained',
              **{f'p{q}_ms': float(np.percentile(timings, q)) for q in (50, 95, 99)}}
    if forward_start is not None:
        result['neural_forward_calls'] = runner.forward_calls - forward_start
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--pgn', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--backend', default='auto')
    parser.add_argument('--runs', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    results = benchmark(load_runner(args.ckpt, device=args.device, backend=args.backend),
                        sample_positions(args.pgn, args.runs, args.seed), args.runs, args.warmup)
    artifact = artifact_info(args.ckpt)
    results.update(checkpoint=args.ckpt, model_bytes=artifact['model_bytes'], model_sha256=artifact['sha256'], seed=args.seed)
    with open(args.pgn, 'rb') as stream:
        results['pgn_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    Path(args.out).write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
