import argparse
import json
import math
from pathlib import Path

import chess

from chessio.records import record_board
from model.artifacts import artifact_info


def harden(row):
    board = record_board(row)
    policy = row['move_probs'] if 'move_probs' in row else {row['move']: 1.}
    if not policy or any(not math.isfinite(p) or p < 0 or chess.Move.from_uci(move) not in board.legal_moves
                         for move, p in policy.items()) or not math.isclose(sum(policy.values()), 1., abs_tol=1e-6):
        raise ValueError('Expected a normalized legal teacher policy')
    best = max(policy.values())
    tied = [move for move, probability in policy.items() if probability == best]
    return {**row, 'move_probs': {move: 1 / len(tied) for move in tied}}


def convert(source, output):
    source, output = Path(source), Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Refusing to replace an existing dataset')
    frozen, count = artifact_info(source), 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.part')
    with source.open() as inputs, temporary.open('x') as saved:
        for count, line in enumerate(inputs, 1):
            saved.write(json.dumps(harden(json.loads(line)), separators=(',', ':')) + '\n')
    if not count or artifact_info(source) != frozen:
        raise ValueError('Empty or changed source')
    temporary.replace(output)
    result = dict(positions=count, source=frozen, artifact=artifact_info(output), producer=artifact_info(__file__),
                  policy='Uniform over exact highest-probability teacher ties; all Q labels and histories are preserved')
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.out), indent=2))
