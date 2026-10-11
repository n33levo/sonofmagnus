import argparse
import json
from pathlib import Path

import chess
import numpy as np

from chessio.mask import argmax_legal_move
from model.artifacts import artifact_info
from play.runner import load_runner


DEPTHS = {'Ruy Lopez': 15, 'Sicilian Defense': 12, "Queen's Gambit": 14,
          "King's Indian Defense": 11, 'French Defense': 13}


def evaluate(runner, records, batch_size=64, depths=DEPTHS):
    if not records or batch_size <= 0:
        raise ValueError('Games and batch size must be positive')
    plies = 2 * max(depths.values())
    correct = np.zeros((len(records), plies), dtype=bool)
    pending = []
    forward_start = getattr(runner, 'forward_calls', None)

    def flush():
        predictions = runner.batch_policy([entry[0] for entry in pending])
        if len(predictions) != len(pending):
            raise ValueError('Missing opening predictions')
        for (board, expected, game, ply), (policy, _) in zip(pending, predictions):
            correct[game, ply] = argmax_legal_move(policy, board) == expected
        pending.clear()

    for game, record in enumerate(records):
        if record['opening'] not in depths or len(record['moves']) < plies:
            raise ValueError('Every game needs a known opening and its complete evaluation prefix')
        board = chess.Board(record.get('root_fen', chess.STARTING_FEN))
        for ply, uci in enumerate(record['moves'][:plies]):
            expected = chess.Move.from_uci(uci)
            if expected not in board.legal_moves:
                raise ValueError('Illegal master-game move')
            pending.append((board.copy(), expected, game, ply))
            board.push(expected)
            if len(pending) == batch_size:
                flush()
    if pending:
        flush()
    rng, results = np.random.default_rng(0), {}
    for family, depth in depths.items():
        indices = [i for i, record in enumerate(records) if record['opening'] == family]
        if not indices:
            continue
        values = correct[indices, :2 * depth]
        game_scores = values.mean(-1)
        boot = game_scores[rng.integers(len(indices), size=(10000, len(indices)))].mean(-1)
        results[family] = {'games': len(indices), 'positions': int(values.size),
            'top1_agreement': float(values.mean()), 'advertised_depth_moves': depth,
            'game_bootstrap_95': np.quantile(boot, [.025, .975]).tolist(),
            'agreement_by_fullmove': correct[indices].reshape(len(indices), -1, 2).mean((0, 2)).tolist()}
    return {'games': len(records), 'positions_evaluated': int(correct.size), 'openings': results,
        'neural_forward_calls': runner.forward_calls - forward_start if forward_start is not None else None,
        'protocol': 'Exact played-master-move agreement through each advertised depth; actual game history, both colours, one prediction per position. Intervals resample whole games.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--games-file', required=True)
    parser.add_argument('--games', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.games_file).read_text().splitlines() if line.strip()]
    if args.games <= 0 or args.games > len(records):
        parser.error('The dataset must contain the requested complete games')
    model, dataset = artifact_info(args.ckpt), artifact_info(args.games_file)
    result = evaluate(load_runner(args.ckpt), records[:args.games], args.batch_size)
    if model != artifact_info(args.ckpt) or dataset != artifact_info(args.games_file):
        raise ValueError('Benchmark artifacts changed during evaluation')
    result.update(model=model, dataset=dataset)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
