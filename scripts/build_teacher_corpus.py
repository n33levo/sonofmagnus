import argparse
import io
import json
import math
from pathlib import Path
import sqlite3

import chess
import zstandard

from chessio.records import holdout_positions
from chessio.remote import RangeReader
from model.artifacts import artifact_info
from train.distill_labeler import cp_to_q_value

URL = 'https://database.lichess.org/lichess_db_eval.jsonl.zst'


def teacher_label(row):
    evaluations = [e for e in row['evals'] if e['depth'] >= 16 and len(e['pvs']) >= 5]
    if not evaluations:
        return None
    board = chess.Board(row['fen'])
    if not board.is_valid() or len(board.piece_map()) < 16:
        return None
    legal = {m.uci() for m in board.legal_moves}
    if len(legal) < 16:
        return None
    for evaluation in sorted(evaluations, key=lambda e: e['depth'], reverse=True):
        scores = {}
        try:
            for pv in evaluation['pvs']:
                move = board.parse_uci(pv['line'].split()[0]).uci()
                cp = pv['cp'] if 'cp' in pv else math.copysign(10000, pv['mate'])
                if move in scores or not math.isfinite(cp):
                    raise ValueError('Invalid teacher action')
                scores[move] = cp if board.turn else -cp
        except (ValueError, KeyError, IndexError):
            continue
        best = sorted(scores, key=scores.get, reverse=True)
        mass = {m: math.exp((scores[m] - scores[best[0]]) / 100.) for m in best[:5]}
        total = sum(mass.values())
        q = {m: cp_to_q_value(cp) for m, cp in scores.items()}
        return {'fen': board.fen(), 'move_probs': {m: p / total for m, p in mass.items()},
                'q_values': q, 'centipawns': scores, 'value': 2 * max(q.values()) - 1,
                'label_depth_min': evaluation['depth'], 'label_depth_max': evaluation['depth'],
                'complete_action_labels': set(scores) == legal, 'weight': 3.,
                'source': 'lichess-deep-multipv'}
    return None


def collect(output, positions, excluded):
    output = Path(output)
    part, database = output.with_suffix('.jsonl.part'), output.with_suffix('.dedup.sqlite')
    if positions < 1 or any(p.exists() for p in (output, part, database, output.with_suffix('.json'))):
        raise ValueError('Positive budget and a new output path required')
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = {str(p): artifact_info(p) for p in excluded}
    blocked, producer = holdout_positions(excluded), artifact_info(__file__)
    count = scanned = heldout = duplicates = complete = 0
    with sqlite3.connect(database) as unique, RangeReader(URL) as response, zstandard.ZstdDecompressor().stream_reader(response) as raw, io.TextIOWrapper(raw) as stream, part.open('x') as saved:
        unique.execute('CREATE TABLE positions (fen TEXT PRIMARY KEY) WITHOUT ROWID')
        for line in stream:
            scanned += 1
            label = teacher_label(json.loads(line))
            if label is None:
                continue
            key = ' '.join(label['fen'].split()[:4])
            if key in blocked:
                heldout += 1
                continue
            if not unique.execute('INSERT OR IGNORE INTO positions VALUES (?)', (key,)).rowcount:
                duplicates += 1
                continue
            saved.write(json.dumps(label, separators=(',', ':')) + '\n')
            count += 1
            complete += label['complete_action_labels']
            if count % 10000 == 0:
                unique.commit()
                saved.flush()
                print(json.dumps({'positions': count, 'scanned': scanned}), flush=True)
            if count == positions:
                break
        headers = response.headers
    if count != positions or artifact_info(__file__) != producer or any(artifact_info(p) != info for p, info in sources.items()):
        raise ValueError('Incomplete corpus or changed inputs')
    part.replace(output)
    database.unlink()
    result = dict(positions=count, source=URL, source_headers=headers, source_records_scanned=scanned,
                  complete_action_rows=complete, depth_min=16, legal_moves_min=16, pieces_min=16,
                  heldout_excluded=heldout, duplicates_excluded=duplicates, excluded_data=sources,
                  artifact=artifact_info(output), producer=producer, license='CC0',
                  source_engine_version='Not recorded by the public archive',
                  labels='Only recorded PV actions are supervised; missing action values stay masked')
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--positions', type=int, default=1000000)
    parser.add_argument('--exclude-data', required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.out, args.positions, args.exclude_data.split(',')), indent=2))
