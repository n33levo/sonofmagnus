import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

import chess
import chess.pgn
import numpy as np

from model.artifacts import artifact_info
from chessio.tls import ssl_context
from scripts.lichess_access import TOKEN_FILE, account


def key(board, move):
    return board.fen(), tuple(m.uci() for m in board.move_stack), move.uci()


def has_rating_update(game, username):
    return any(player.get('user', {}).get('id', '').lower() == username.lower()
               and type(player.get('ratingDiff')) is int for player in game['players'].values())


def rated_game_count(username, since, token=None):
    token = token or TOKEN_FILE.read_text().strip()
    query = urllib.parse.urlencode({'since': since, 'until': int(time.time() * 1000),
                                   'rated': 'true', 'perfType': 'blitz', 'ongoing': 'false'})
    url = 'https://lichess.org/api/games/user/' + urllib.parse.quote(username, safe='') + '?' + query
    with urllib.request.urlopen(urllib.request.Request(url, headers={
            'Accept': 'application/x-ndjson', 'Authorization': 'Bearer ' + token}), timeout=30, context=ssl_context()) as response:
        games = [json.loads(line) for line in response.read().splitlines() if line.strip()]
    return sum(item['rated'] and item['variant'] == 'standard' and item['speed'] == 'blitz'
               and item['status'] not in ('created', 'started', 'aborted')
               and has_rating_update(item, username) for item in games)


def retained_aborts(directory, exported_ids):
    result = []
    for path in sorted(Path(directory).glob('*.pgn')):
        game = chess.pgn.read_game(path.open())
        if game is None or game.errors:
            raise ValueError('Invalid retained game PGN')
        headers = game.headers
        if headers['Result'] != '*' or headers.get('Termination') != 'Abandoned':
            continue
        identifier = headers['GameId']
        if identifier in exported_ids:
            continue
        board, moves = game.board(), []
        for move in game.mainline_moves():
            moves.append(board.san(move))
            board.push(move)
        clock, increment = map(int, headers['TimeControl'].split('+'))
        start = datetime.strptime(headers['UTCDate'] + ' ' + headers['UTCTime'], '%Y.%m.%d %H:%M:%S').replace(tzinfo=timezone.utc)
        result.append({'id': identifier, 'status': 'aborted', 'rated': False, 'variant': 'standard', 'speed': 'blitz',
            'players': {color: {'user': {'id': headers[color.title()].lower()}} for color in ('white', 'black')},
            'clock': {'initial': clock, 'increment': increment}, 'createdAt': int(start.timestamp() * 1000),
            'lastMoveAt': int(start.timestamp() * 1000), 'moves': ' '.join(moves), 'pgn': path.read_text(),
            'source': 'Locally retained Lichess PGN; aborted games are omitted from the bulk API export',
            'source_artifact': artifact_info(path)})
    return result


def review(games, records, username, *, played_records=None):
    decisions = defaultdict(deque)
    for row in records:
        board = chess.Board(row['root_fen'])
        for move in row['history']:
            board.push_uci(move)
        if board.fen() != row['fen']:
            raise ValueError('Audit history does not reconstruct the position')
        if row['move'] is not None:
            move = board.parse_uci(row['move'])
            if row['neural_forward_calls'] != 1 or not np.isfinite(row['latency_ms']) or row['latency_ms'] < 0:
                raise ValueError('Invalid neural call count or latency')
            decisions[key(board, move)].append(row)
    starts = sorted(item['createdAt'] for item in games if 'createdAt' in item)
    cancelled = []
    def belongs(row, item):
        if 'createdAt' not in item:
            return True
        if 'utc_seconds' not in row:
            return False
        end = next((start for start in starts if start > item['createdAt']), float('inf'))
        return item['createdAt'] <= 1000 * row['utc_seconds'] < end
    def consume(board, move, item):
        candidates = decisions[key(board, move)]
        row = next((row for row in candidates if belongs(row, item)), None)
        if row is None:
            raise ValueError(f'Missing neural decision for {item["id"]} at ply {len(board.move_stack)}')
        candidates.remove(row)
        return row
    def cancel(row, item, reason):
        cancelled.append({'game': item['id'], 'reason': reason, 'pid': row.get('pid'),
                          'history_plies': len(row['history']), 'utc_seconds': row.get('utc_seconds')})
    outcomes, timings, reviewed, plies, ids, aborted = [0, 0, 0], [], [], 0, set(), []
    for item in games:
        if item['id'] in ids:
            raise ValueError('Repeated server game ID')
        ids.add(item['id'])
        abandoned = item['status'] == 'aborted'
        if item['status'] in ('created', 'started'):
            raise ValueError('Game is still running')
        if (not abandoned and not item['rated']) or item['variant'] != 'standard' or item['speed'] != 'blitz':
            raise ValueError('Expected rated standard blitz')
        sides = {color: item['players'][color]['user']['id'].lower() for color in ('white', 'black')}
        if username.lower() not in sides.values():
            raise ValueError('Game belongs to another account')
        color = chess.WHITE if sides['white'] == username.lower() else chess.BLACK
        game = chess.pgn.read_game(io.StringIO(item['pgn']))
        if game is None or game.errors:
            raise ValueError('Invalid server PGN')
        server_board = game.board()
        for text in item['moves'].split():
            server_board.push_san(text)
        if list(game.mainline_moves()) != server_board.move_stack:
            raise ValueError('Server move list and PGN differ')
        expected = '*' if abandoned else '1-0' if item.get('winner') == 'white' else '0-1' if item.get('winner') == 'black' else '1/2-1/2'
        if game.headers['Result'] != expected:
            raise ValueError('PGN and server outcome differ')
        board, own_moves = game.board(), 0
        for move in game.mainline_moves():
            if move not in board.legal_moves:
                raise ValueError('Illegal server game move')
            if board.turn == color:
                row = consume(board, move, item)
                if abandoned:
                    cancel(row, item, 'move in aborted game; excluded from rated results')
                else:
                    timings.append(row['latency_ms'])
                    if played_records is not None:
                        played_records.append((item['id'], row))
                own_moves += 1
            board.push(move)
            plies += not abandoned
        # Asynchronous game-over notifications can cancel an already requested move.
        # Match only the exact final history in the game's time window.
        if board.turn == color:
            for move in board.legal_moves:
                candidates = decisions[key(board, move)]
                for row in list(candidates):
                    if belongs(row, item) and 'utc_seconds' in row and 1000 * row['utc_seconds'] >= item.get('lastMoveAt', float('inf')):
                        candidates.remove(row)
                        cancel(row, item, 'uncommitted request at finished game root')
        if abandoned:
            aborted.append({'id': item['id'], 'source': item.get('source', 'server export'), 'neural_moves': own_moves})
            continue
        if item['status'] == 'mate' and not board.is_checkmate():
            raise ValueError('Server mate does not replay as checkmate')
        winner = item.get('winner')
        score = .5 if winner is None else float(winner == ('white' if color else 'black'))
        outcomes[0 if score == 1 else 1 if score == .5 else 2] += 1
        reviewed.append({'id': item['id'], 'url': 'https://lichess.org/' + item['id'], 'score': score,
                         'opponent': sides['black' if color else 'white'], 'clock': item['clock'],
                         'status': item['status'], 'neural_moves': own_moves,
                         'rating_updated': has_rating_update(item, username)})
    return {'games': len(reviewed), 'wins': outcomes[0], 'draws': outcomes[1], 'losses': outcomes[2],
            'legal_plies': plies, 'neural_moves_verified': len(timings), 'games_reviewed': reviewed,
            'rating_updates': sum(item['rating_updated'] for item in reviewed),
            'games_without_rating_update': [item['id'] for item in reviewed if not item['rating_updated']],
            'aborted_games': aborted, 'aborted_game_ids': [item['id'] for item in aborted],
            'cancelled_decisions': cancelled, 'unmatched_decisions': sum(len(rows) for rows in decisions.values()),
            'latency_ms': dict(zip(('p50', 'p95', 'p99'), np.percentile(timings, [50, 95, 99]).tolist())) if timings else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay rated Lichess games against the local neural decision log')
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    root = args.run / manifest['runtime_directory'] if 'runtime_directory' in manifest else Path(__file__).resolve().parents[1]
    if any(artifact_info(root / path)['sha256'] != digest for path, digest in manifest['runtime_source'].items()):
        raise ValueError('Runtime source differs from the online run')
    model = args.run / Path(manifest['source_checkpoint']).name
    if artifact_info(model) != manifest['model']:
        raise ValueError('Frozen online model changed')
    records = [json.loads(line) for line in (args.run / 'moves.jsonl').read_text().splitlines()]
    # The launch file was frozen before this older manifest acquired timestamps.
    since = manifest.get('started_ms', int((args.run / 'bot.py').stat().st_mtime * 1000))
    username = manifest['account']
    token = TOKEN_FILE.read_text().strip()
    profile = account(token, username)
    query = urllib.parse.urlencode({'since': since, 'until': int(time.time() * 1000), 'perfType': 'blitz',
        'pgnInJson': 'true', 'clocks': 'true', 'evals': 'false', 'ongoing': 'false', 'sort': 'dateAsc'})
    url = 'https://lichess.org/api/games/user/' + urllib.parse.quote(username, safe='') + '?' + query
    with urllib.request.urlopen(urllib.request.Request(url, headers={
            'Accept': 'application/x-ndjson', 'Authorization': 'Bearer ' + token}), timeout=60, context=ssl_context()) as response:
        payload = response.read()
    games = [json.loads(line) for line in payload.splitlines() if line.strip()]
    (args.run / 'server_games.jsonl').write_bytes(payload)
    aborts = retained_aborts(args.run / 'games', {item['id'] for item in games})
    games = sorted([*games, *aborts], key=lambda item: item.get('createdAt', 0))
    result = review(games, records, username)
    result.update(account=username, blitz=profile.get('perfs', {}).get('blitz'), model=manifest['model'],
                  server_games=artifact_info(args.run / 'server_games.jsonl'), selection_start_ms=since,
                  reviewer=artifact_info(__file__), source_hashes_verified=True,
                  rating_context='Actual Lichess BOT-account blitz rating; inspect provisional flag and RD. Not FIDE Elo.',
                  retained_aborted_pgns={item['id']: item['source_artifact'] for item in aborts},
                  status='Completed games independently replayed; known aborted-game and finished-root requests reported separately; unknown decisions remain unmatched')
    (args.run / 'review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
