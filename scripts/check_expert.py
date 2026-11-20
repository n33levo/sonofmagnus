import argparse
import json
from pathlib import Path
import sqlite3
import tempfile

import chess

from chessio.records import holdout_positions
from model.artifacts import artifact_info


def check(path):
    path = Path(path)
    manifest = json.loads(path.with_suffix('.json').read_text())
    frozen = artifact_info(path)
    assert frozen == manifest['artifact']
    assert artifact_info('scripts/fetch_expert.py') == manifest['producer']
    for source, expected in manifest['excluded_data'].items():
        assert artifact_info(source) == expected
    blocked = holdout_positions(manifest['excluded_data'])
    count, phases = 0, dict.fromkeys(('0', '1', '2', '3'), 0)
    with tempfile.TemporaryDirectory() as directory, sqlite3.connect(Path(directory) / 'unique.sqlite') as unique, path.open() as stream:
        unique.execute('CREATE TABLE positions (fen TEXT PRIMARY KEY) WITHOUT ROWID')
        for line in stream:
            row = json.loads(line)
            assert set(row) == {'fen', 'move', 'weight', 'source'}
            assert row['weight'] == 3 and row['source'] == 'chessbench-stockfish-top1'
            board = chess.Board(row['fen'])
            assert board.is_valid() and chess.Move.from_uci(row['move']) in board.legal_moves
            key = ' '.join(board.fen().split()[:4])
            assert key not in blocked
            assert unique.execute('INSERT OR IGNORE INTO positions VALUES (?)', (key,)).rowcount == 1
            pieces = len(board.piece_map())
            phases[str(0 if pieces >= 24 else 1 if pieces >= 16 else 2 if pieces >= 8 else 3)] += 1
            count += 1
            if count % 100000 == 0:
                unique.commit()
                print(json.dumps({'reviewed': count}), flush=True)
    assert count == manifest['positions'] and phases == manifest['pieces_phases']
    assert artifact_info(path) == frozen
    return dict(positions=count, legal_teacher_moves=count, distinct_positions=True, frozen_tests_excluded=True,
                pieces_phases=phases, artifact=frozen, reviewer=artifact_info(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = check(args.source)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
