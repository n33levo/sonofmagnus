"""Download and replay n33levo's completed public Lichess games."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import urllib.request

import chess.pgn

from model.artifacts import artifact_info


ACCOUNT = 'n33levo'
URL = f'https://lichess.org/api/games/user/{ACCOUNT}?clocks=true&evals=true&opening=true&pgnInJson=true&accuracy=true'


def replay(record):
    if record['variant'] != 'standard' or record['status'] in ('created', 'started', 'aborted'):
        raise ValueError('Expected a completed standard game')
    colors = [color for color in ('white', 'black') if record['players'][color].get('user', {}).get('id') == ACCOUNT]
    if len(colors) != 1:
        raise ValueError('Account must occupy exactly one side')
    color = colors[0]
    score = .5 if not record.get('winner') else float(record['winner'] == color)
    expected = '1/2-1/2' if score == .5 else '1-0' if record['winner'] == 'white' else '0-1'
    game = chess.pgn.read_game(io.StringIO(record['pgn']))
    if game is None or game.errors or game.headers['Result'] != expected:
        raise ValueError('PGN and server result disagree')
    board, history, positions = game.board(), [], []
    root = board.fen()
    moves = list(game.mainline_moves())
    san = record.get('moves', '').split()
    clocks = record.get('clocks', [])
    if len(moves) != len(san) or (clocks and len(clocks) not in (len(moves), len(moves) + 1)):
        raise ValueError('PGN, SAN or clock lengths disagree')
    for node, clock in zip(game.mainline(), clocks):
        if node.clock() is not None and abs(node.clock() * 100 - clock) > 51:
            raise ValueError('PGN and JSON clocks disagree')
    for ply, (move, text) in enumerate(zip(moves, san)):
        if move not in board.legal_moves or board.parse_san(text) != move:
            raise ValueError('Illegal or mismatched game history')
        if board.turn == (color == 'white'):
            positions.append(dict(game_id=record['id'], ply=ply, own_move=len(positions) + 1,
                root_fen=root, history=list(history), fen=board.fen(), played_move=move.uci(),
                clock_after_cs=clocks[ply] if clocks else None))
        board.push(move)
        history.append(move.uci())
    own, opponent = record['players'][color], record['players']['black' if color == 'white' else 'white']
    summary = dict(game_id=record['id'], url='https://lichess.org/' + record['id'],
        created_ms=record['createdAt'], finished_ms=record['lastMoveAt'], status=record['status'],
        rated=record['rated'], perf=record['perf'], color=color, score=score,
        own_rating=own['rating'], opponent_rating=opponent['rating'],
        opponent=opponent.get('user', {}).get('id'), opponent_title=opponent.get('user', {}).get('title'),
        rating_diff=own.get('ratingDiff'), clock=record.get('clock'), own_moves=len(positions),
        final_fen=board.fen(), opening=record.get('opening', {}), extra_terminal_clock=len(clocks) == len(moves) + 1)
    return summary, positions


def fetch(folder):
    folder.mkdir(parents=True, exist_ok=True)
    data, manifest = folder / 'games.ndjson', folder / 'manifest.json'
    if data.exists():
        saved = json.loads(manifest.read_text())
        actual = artifact_info(data)
        if actual['sha256'] != saved['sha256']:
            raise ValueError('Downloaded game snapshot changed')
        return
    request = urllib.request.Request(URL, headers={'Accept': 'application/x-ndjson', 'User-Agent': 'sonofmagnus-local-benchmark'})
    rows = []
    with urllib.request.urlopen(request, timeout=60) as response:
        for line in response:
            if line.strip():
                row = json.loads(line)
                if 'id' not in row or 'players' not in row:
                    raise ValueError('Unexpected Lichess export response')
                rows.append(row)
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Empty or duplicated game export')
    temporary = data.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))
    temporary.replace(data)
    manifest.write_text(json.dumps(dict(account=ACCOUNT, url=URL, downloaded_utc=datetime.now(timezone.utc).isoformat(),
        games=len(rows), sha256=artifact_info(data)['sha256'], bytes=data.stat().st_size,
        automatic_rated=False), indent=2) + '\n')


def prepare(folder):
    fetch(folder)
    data = folder / 'games.ndjson'
    records = sorted(map(json.loads, data.open()), key=lambda row: (row['createdAt'], row['id']))
    if len(records) != len({row['id'] for row in records}):
        raise ValueError('Duplicate game IDs')
    games, positions = [], []
    for record in records:
        game, roots = replay(record)
        games.append(game)
        positions.extend(roots)
    for name, rows in [('games.jsonl', games), ('positions.jsonl', positions)]:
        (folder / name).write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))
    outcomes = Counter('win' if row['score'] == 1 else 'loss' if row['score'] == 0 else 'draw' for row in games)
    report = dict(account=ACCOUNT, games=len(games), outcomes=dict(outcomes),
        score=sum(row['score'] for row in games) / len(games), own_positions=len(positions),
        rating_change_recorded=sum(row['rating_diff'] is not None for row in games),
        no_recorded_rating_change=[row['game_id'] for row in games if row['rating_diff'] is None],
        server_analysis_games=sum(bool(row.get('analysis')) for row in records),
        inputs={str(data): artifact_info(data), str(Path(__file__)): artifact_info(__file__)},
        artifacts={name: artifact_info(folder / name) for name in ('games.jsonl', 'positions.jsonl')},
        scope='All exported completed games, with every own-move history replayed independently against PGN and SAN. '
              'Records without rating changes are retained separately. This is a retrospective mixed-model benchmark, not untouched Elo evidence.',
        automatic_rated=False, training_allowed=False)
    (folder / 'history-review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('data/lichess-history'))
    prepare(parser.parse_args().out)
