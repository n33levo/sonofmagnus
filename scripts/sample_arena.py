import argparse
import json
from pathlib import Path
import random

import chess

from chessio.records import holdout_positions
from model.artifacts import artifact_info


def sample(source, output, count=500, seed=0, exclude=(), previous=()):
    source, output = Path(source), Path(output)
    if count < 1 or output.exists():
        raise ValueError('Use a positive count and new output path')
    frozen = artifact_info(source)
    provenance = json.loads(source.with_suffix('.json').read_text())
    if provenance['dataset'] != frozen:
        raise ValueError('Master-game sample differs from its manifest')
    blocked = holdout_positions(exclude)
    for path in previous:
        blocked.update(' '.join(line.split()[:4]) for line in Path(path).read_text().splitlines())
    positions, rng, games = {}, random.Random(seed), 0
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            board, target = chess.Board(row['root_fen']), rng.randint(8, 24)
            for ply, move in enumerate(row['moves'], 1):
                board.push_uci(move)
                if ply == target:
                    key = ' '.join(board.fen().split()[:4])
                    if key not in blocked and not board.is_game_over(claim_draw=True):
                        positions.setdefault(key, board.fen())
                    break
            games += 1
    if games != provenance['games'] or len(positions) < count or artifact_info(source) != frozen:
        raise ValueError('Wrong game count, insufficient roots or changed source')
    selected = rng.sample(sorted(positions.values()), count)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write('\n'.join(selected) + '\n')
    result = dict(positions=count, source_games=games, eligible_distinct_positions=len(positions),
                  source=frozen, artifact=artifact_info(output), seed=seed, plies=[8, 24],
                  excluded_inputs=list(exclude), earlier_roots=list(previous),
                  source_urls=provenance['source_urls'], license=provenance['license'],
                  producer=artifact_info(__file__))
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exclude-data', default='')
    parser.add_argument('--previous', default='')
    args = parser.parse_args()
    print(json.dumps(sample(args.source, args.out, args.count, args.seed,
                            args.exclude_data.split(',') if args.exclude_data else (),
                            args.previous.split(',') if args.previous else ()), indent=2))
