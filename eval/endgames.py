import argparse
from collections import defaultdict
import json
from pathlib import Path

import chess

from play.runner import load_runner


def evaluate_endgames(runner, labels_path):
    buckets = defaultdict(lambda: {'total': 0, 'optimal': 0, 'absolute_value_error': 0., 'forced_choice_total': 0, 'forced_choice_optimal': 0})
    with open(labels_path) as stream:
        for line in stream:
            item = json.loads(line)
            board = chess.Board(item['fen'])
            logits, value = runner.policy(board)
            from chessio.mask import argmax_legal_move
            move = argmax_legal_move(logits, board).uci()
            targets = item['q_values']
            best = max(targets.values())
            optimal = targets[move] == best
            bucket = buckets[len(board.piece_map())]
            bucket['total'] += 1
            bucket['optimal'] += optimal
            bucket['absolute_value_error'] += abs(value - item['value'])
            if min(targets.values()) < best:
                bucket['forced_choice_total'] += 1
                bucket['forced_choice_optimal'] += optimal
    if not buckets:
        raise ValueError('Empty endgame holdout')
    return {**{pieces: {**stats, 'optimal_move_rate': stats['optimal'] / stats['total'],
                       'value_mae': stats['absolute_value_error'] / stats['total']} for pieces, stats in buckets.items()},
            'protocol': 'Preserve tablebase WDL under the fifty-move rule; DTZ tie-breaking is not scored'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out')
    args = parser.parse_args()
    results = evaluate_endgames(load_runner(args.ckpt, device=args.device), args.labels)
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
