import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path

from chessio.records import record_board
from model.artifacts import artifact_info


def check(source, labels):
    source, labels = Path(source), Path(labels)
    manifest = json.loads(labels.with_suffix('.json').read_text())
    source_hash, label_hash = artifact_info(source), artifact_info(labels)
    if manifest['status'] != 'complete' or manifest['identity']['source'] != source_hash or manifest['artifact'] != label_hash:
        raise ValueError('Input or labels differ from the completed manifest')
    depth, count = manifest['identity']['depth'], 0
    with source.open() as inputs, labels.open() as output:
        for original, labeled in zip_longest(inputs, output):
            if original is None or labeled is None:
                raise ValueError('Input and label counts differ')
            row, target = json.loads(original), json.loads(labeled)
            for field in ('fen', 'root_fen', 'history', 'game_id', 'move', 'predicted_value', 'policy_entropy'):
                if row.get(field) != target.get(field):
                    raise ValueError(f'Input {field} changed')
            board = record_board(target)
            legal = {m.uci() for m in board.legal_moves}
            q, policy = target['q_values'], target['move_probs']
            if not board.is_valid() or set(q) != legal or not set(policy) <= legal:
                raise ValueError('Invalid board or incomplete legal-action labels')
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in [*q.values(), *policy.values()]):
                raise ValueError('Invalid probability')
            if not math.isclose(sum(policy.values()), 1., abs_tol=1e-10):
                raise ValueError('Policy mass differs from one')
            if not math.isclose(target['value'], 2 * max(q.values()) - 1, abs_tol=1e-10):
                raise ValueError('Side-to-move value differs from action values')
            if not depth <= target['label_depth_min'] <= target['label_depth_max']:
                raise ValueError('Labels below requested depth')
            count += 1
    if count != manifest['positions'] or artifact_info(source) != source_hash or artifact_info(labels) != label_hash:
        raise ValueError('Wrong count or changed artifact')
    return dict(positions=count, source=source_hash, labels=label_hash, minimum_depth=depth,
                histories_verified=True, all_legal_actions_verified=True,
                finite_probability_mass_verified=True, side_to_move_value_verified=True,
                reviewer_sha256=artifact_info(__file__)['sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = check(args.source, args.labels)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
