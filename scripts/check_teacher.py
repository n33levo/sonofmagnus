import argparse
from collections import Counter
import json
import math
from pathlib import Path

import chess

from chessio.records import holdout_positions
from model.artifacts import artifact_info


def review(path, excluded, expected):
    frozen = artifact_info(path)
    blocked, seen, moves, depths, coverage = holdout_positions(excluded), set(), Counter(), Counter(), Counter()
    for line in Path(path).open():
        row = json.loads(line)
        board = chess.Board(row['fen'])
        key = ' '.join(board.fen().split()[:4])
        legal = {move.uci() for move in board.legal_moves}
        if not board.is_valid() or len(board.piece_map()) < 16 or len(legal) < 16 or key in seen or key in blocked:
            raise ValueError('Invalid, repeated or held-out position')
        seen.add(key)
        q, cp, policy = row['q_values'], row['centipawns'], row['move_probs']
        if len(q) < 5 or set(q) != set(cp) or not set(q) <= legal or row['complete_action_labels'] != (set(q) == legal):
            raise ValueError('Incorrect action coverage')
        for move, score in cp.items():
            z = score / 400
            expected_q = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
            if not math.isfinite(score) or not math.isfinite(q[move]) or abs(q[move] - expected_q) > 1e-12:
                raise ValueError('Incorrect score transformation')
        mass = {move: math.exp((cp[move] - max(cp.values())) / 100) for move in sorted(cp, key=cp.get, reverse=True)[:5]}
        if set(policy) != set(mass) or any(not math.isfinite(policy[m]) or abs(policy[m] - p / sum(mass.values())) > 1e-12 for m, p in mass.items()):
            raise ValueError('Incorrect teacher policy')
        if not math.isfinite(row['value']) or abs(row['value'] - (2 * max(q.values()) - 1)) > 1e-12 or row['label_depth_min'] < 16 or row['label_depth_max'] < row['label_depth_min']:
            raise ValueError('Incorrect value or depth')
        moves[len(legal)] += 1
        depths[row['label_depth_min']] += 1
        coverage[len(q)] += 1
    if len(seen) != expected or artifact_info(path) != frozen:
        raise ValueError('Incomplete or changed corpus')
    return {'positions': len(seen), 'minimum_legal_moves': min(moves), 'minimum_depth': min(depths),
            'observed_action_counts': dict(coverage), 'all_holdouts_excluded': True,
            'independent_transform_and_policy_checks': True, 'artifact': frozen}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--exclude-data', required=True)
    parser.add_argument('--positions', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = review(args.source, args.exclude_data.split(','), args.positions)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
