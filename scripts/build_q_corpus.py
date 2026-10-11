import argparse
import io
import json
import math
from itertools import islice
from pathlib import Path
import sqlite3

import chess
import zstandard

from chessio.remote import RangeReader
from model.artifacts import artifact_info
from train.distill_labeler import cp_to_q_value, centipawns_to_value


def complete_label(row, depth=12):
    evaluations = [e for e in row['evals'] if e['depth'] >= depth and len(e['pvs']) >= 2]
    if not evaluations:
        return None
    board = chess.Board(row['fen'])
    maximum = max(len(e['pvs']) for e in evaluations)
    legal = set(islice(board.legal_moves, maximum + 1))
    if len(legal) > maximum or len(legal) < 2 or not board.is_valid():
        return None
    for evaluation in sorted(evaluations, key=lambda e: e['depth'], reverse=True):
        if len(evaluation['pvs']) != len(legal):
            continue
        scored = {}
        try:
            for pv in evaluation['pvs']:
                move = board.parse_uci(pv['line'].split()[0]).uci()
                cp = pv['cp'] if 'cp' in pv else math.copysign(10000, pv['mate'])
                if not math.isfinite(cp) or move in scored:
                    raise ValueError('Invalid or repeated action value')
                scored[move] = cp if board.turn else -cp
        except (ValueError, KeyError, IndexError):
            continue
        if set(scored) != {move.uci() for move in legal}:
            continue
        best = sorted(scored, key=scored.get, reverse=True)
        mass = {move: math.exp((scored[move] - scored[best[0]]) / 100.) for move in best[:5]}
        total = sum(mass.values())
        return {'fen': board.fen(), 'move_probs': {move: probability / total for move, probability in mass.items()},
            'q_values': {move: cp_to_q_value(cp) for move, cp in scored.items()},
            'value': centipawns_to_value(scored[best[0]]), 'weight': 3.,
            'label_depth_min': evaluation['depth'], 'label_depth_max': evaluation['depth']}
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--positions', type=int, default=2500000)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.positions <= 0:
        parser.error('Positive position count required')
    out = Path(args.out)
    if out.exists() or out.with_suffix('.part').exists():
        parser.error('Use a new output path')
    out.parent.mkdir(parents=True, exist_ok=True)
    blocked = set()
    for name in ('puzzles_holdout', 'puzzles_final3000', 'stockfish_holdout', 'eval_holdout', 'endgame_holdout'):
        for line in Path('data', name + '.jsonl').read_text().splitlines():
            row = json.loads(line)
            board = chess.Board(row['fen'])
            blocked.add(' '.join(board.fen().split()[:4]))
            for move in row.get('moves', []):
                board.push_uci(move)
                blocked.add(' '.join(board.fen().split()[:4]))
    url = 'https://database.lichess.org/lichess_db_eval.jsonl.zst'
    database = out.with_suffix('.dedup.sqlite')
    positions = scanned = excluded = duplicates = 0
    with sqlite3.connect(database) as unique, RangeReader(url) as response, zstandard.ZstdDecompressor().stream_reader(response) as raw, io.TextIOWrapper(raw) as stream, out.with_suffix('.part').open('w') as saved:
        unique.execute('CREATE TABLE IF NOT EXISTS positions (fen TEXT PRIMARY KEY) WITHOUT ROWID')
        for line in stream:
            scanned += 1
            label = complete_label(json.loads(line))
            if label is None:
                continue
            key = ' '.join(label['fen'].split()[:4])
            if key in blocked:
                excluded += 1
                continue
            if not unique.execute('INSERT OR IGNORE INTO positions VALUES (?)', (key,)).rowcount:
                duplicates += 1
                continue
            saved.write(json.dumps(label, separators=(',', ':')) + '\n')
            positions += 1
            if positions % 10000 == 0:
                unique.commit()
                print(json.dumps({'positions': positions, 'scanned': scanned}), flush=True)
            if positions == args.positions:
                break
        headers = response.headers
    if positions != args.positions:
        raise ValueError('Archive ended before the requested full-action label count')
    out.with_suffix('.part').replace(out)
    database.unlink()
    result = {'positions': positions, 'source_records_scanned': scanned, 'heldout_positions_excluded': excluded,
        'duplicates_excluded': duplicates, 'depth_min': 12, 'legal_moves_min': 2, 'source': url,
        'source_headers': headers, 'license': 'CC0', 'score_perspective': 'White source scores converted to side-to-move',
        'source_engine_version': 'Not recorded by the public archive', 'artifact': artifact_info(out)}
    out.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
