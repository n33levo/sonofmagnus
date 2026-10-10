import argparse
import json
import math
from pathlib import Path

import chess
import chess.engine
import numpy as np


def cp_to_q_value(cp):
    z = cp / 400.
    return 1. / (1. + math.exp(-z)) if z >= 0 else math.exp(z) / (1. + math.exp(z))


def centipawns_to_value(cp):
    return 2 * cp_to_q_value(cp) - 1


def label_position(board, engine, depth=14, time_limit=.1, top_k=5, include_q_values=False):
    legal = set(board.legal_moves)
    if not legal:
        raise ValueError('Cannot label a terminal position')
    infos = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit or None),
                           multipv=len(legal) if include_q_values else min(top_k, len(legal)))
    scored = []
    for info in infos:
        if not info.get('pv'):
            continue
        move = info['pv'][0]
        cp = info['score'].pov(board.turn).score(mate_score=10000)
        if move not in legal or cp is None:
            raise ValueError('Engine returned an invalid label')
        scored.append((move, cp, info.get('depth', 0)))
    scored.sort(key=lambda item: item[1], reverse=True)
    if not scored or (include_q_values and {m for m, _, _ in scored} != legal):
        raise ValueError('Engine did not label every requested legal move')
    scores = np.array([cp for _, cp, _ in scored[:top_k]], dtype=np.float64) / 100.
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    result = {'fen': board.fen(), 'move_probs': {m.uci(): float(p) for (m, _, _), p in zip(scored, probs)},
              'value': centipawns_to_value(scored[0][1]),
              'label_depth_min': min(d for _, _, d in scored), 'label_depth_max': max(d for _, _, d in scored)}
    if include_q_values:
        result['q_values'] = {m.uci(): cp_to_q_value(cp) for m, cp, _ in scored}
    return result


def label_positions_from_file(input_path, output_path, engine_path='stockfish', depth=14,
                              time_limit=.1, top_k=5, max_positions=None, include_q_values=False):
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError('Input and output must differ')
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    count = 0
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine, open(input_path) as source, temporary.open('w') as out:
            engine.configure({'Threads': 1, 'Hash': 128})
            for line in source:
                if max_positions is not None and count >= max_positions:
                    break
                item = json.loads(line)
                board = chess.Board(item.get('fen', item.get('position')))
                if board.is_game_over(claim_draw=True):
                    continue
                label = label_position(board, engine, depth, time_limit, top_k, include_q_values)
                out.write(json.dumps({**item, **label}) + '\n')
                count += 1
                if count % 100 == 0:
                    print(f'Labeled {count} positions', flush=True)
        if not count:
            raise ValueError('No positions labeled')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Labeled {count} positions -> {output}')


def main():
    parser = argparse.ArgumentParser(description='Offline Stockfish labels')
    parser.add_argument('--fens', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--engine', default='stockfish')
    parser.add_argument('--depth', type=int, default=14)
    parser.add_argument('--time', type=float, default=.1, help='0 means no time cap')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--max', type=int)
    parser.add_argument('--preset', choices=['fast', 'deep'])
    parser.add_argument('--q-values', action='store_true')
    args = parser.parse_args()
    if args.preset:
        args.depth, args.time, args.topk = (12, .05, 5) if args.preset == 'fast' else (18, .5, 12)
    if args.topk <= 0 or args.depth <= 0 or args.time < 0:
        parser.error('Depth and topk must be positive; time must be nonnegative')
    label_positions_from_file(args.fens, args.out, args.engine, args.depth, args.time,
                              args.topk, args.max, args.q_values)


if __name__ == '__main__':
    main()
