import argparse
from collections import Counter
import json
from pathlib import Path
import random

import chess

from chessio.records import holdout_positions
from model.artifacts import artifact_info


def game_positions(rows, blocked, seed=0):
    rng = random.Random(seed)
    current, board, chosen, eligible = None, None, None, 0
    valid = False
    for row in rows:
        if row['game_id'] != current:
            if valid and chosen:
                yield chosen
            current, chosen, eligible = row['game_id'], None, 0
            valid = row.get('weight', 0) in (2, 3) and row['fen'] == chess.STARTING_FEN
            board = chess.Board() if valid else None
        if not valid:
            continue
        if board.fen() != row['fen']:
            valid, chosen = False, None
            continue
        move = chess.Move.from_uci(row['move'])
        if move not in board.legal_moves:
            raise ValueError('Illegal move in the human corpus')
        ply = len(board.move_stack)
        if 16 <= ply <= 120 and ' '.join(board.fen().split()[:4]) not in blocked and not board.is_game_over(claim_draw=True):
            eligible += 1
            if rng.randrange(eligible) == 0:
                chosen = {**row, 'root_fen': chess.STARTING_FEN,
                          'history': [m.uci() for m in board.move_stack], 'source': 'lichess-master-history'}
        board.push(move)
    if valid and chosen:
        yield chosen


def collect(paths, output, count, excluded, seed=0):
    paths, output = [Path(p) for p in paths], Path(output)
    part = output.with_suffix('.jsonl.part')
    if count < 1 or len(set(p.resolve() for p in paths)) != len(paths) or any(p.exists() for p in (output, part, output.with_suffix('.json'))):
        raise ValueError('Positive budget, distinct sources and a new output path required')
    sources = {str(p): artifact_info(p) for p in paths}
    for path in paths:
        manifest = json.loads(path.with_suffix('.json').read_text())
        if manifest['artifact'] != sources[str(path)] or manifest['rating_range'] != [2000, 2800]:
            raise ValueError('Source is not the verified human rating-tier corpus')
    tests = {str(p): artifact_info(p) for p in excluded}
    blocked = holdout_positions(excluded)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen, phases, positions = set(), Counter(), 0

    def rows():
        for path in paths:
            with path.open() as source:
                for line in source:
                    yield json.loads(line)

    with part.open('x') as saved:
        for row in game_positions(rows(), blocked, seed):
            if row['game_id'] in seen:
                continue
            seen.add(row['game_id'])
            saved.write(json.dumps(row, separators=(',', ':')) + '\n')
            board = chess.Board(row['fen'])
            pieces = len(board.piece_map())
            phases['24+' if pieces >= 24 else '16-23' if pieces >= 16 else '8-15' if pieces >= 8 else '2-7'] += 1
            positions += 1
            if positions % 1000 == 0:
                saved.flush()
                print(json.dumps({'positions': positions, 'distinct_games': len(seen)}), flush=True)
            if positions == count:
                break
    if positions != count or any(artifact_info(p) != frozen for p, frozen in {**sources, **tests}.items()):
        raise ValueError('Incomplete corpus or changed sources')
    part.replace(output)
    result = dict(positions=positions, distinct_games=len(seen), sources=sources, excluded_data=tests,
                  seed=seed, rating_min=2400, rating_basis='Collector weight tiers 2 and 3; both players were 2400-2800',
                  pieces_phases=dict(phases), plies=[16, 120], history='Verified contiguous moves from the real starting position; games with gaps are dropped',
                  selection='One uniformly sampled eligible position per contiguous game prefix, in source order',
                  artifact=artifact_info(output), producer=artifact_info(__file__), license='CC0')
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--positions', type=int, default=20000)
    parser.add_argument('--exclude-data', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(collect(args.data.split(','), args.out, args.positions,
                             args.exclude_data.split(','), args.seed), indent=2))
