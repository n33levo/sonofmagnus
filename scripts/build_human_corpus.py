import argparse
import io
import json
from contextlib import ExitStack
from pathlib import Path

import chess
import chess.pgn
import zstandard

from model.artifacts import artifact_info
from chessio.remote import RangeReader


class Rated(chess.pgn.GameBuilder):
    def end_headers(self):
        headers = self.game.headers
        try:
            ratings = [int(headers.get(color + 'Elo', 0)) for color in ('White', 'Black')]
        except ValueError:
            return chess.pgn.SKIP
        if min(ratings) < 2000 or max(ratings) > 2800 or headers.get('Event') not in ('Rated Blitz game', 'Rated Rapid game'):
            return chess.pgn.SKIP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--month', required=True)
    parser.add_argument('--positions', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume', action='store_true', help='Replay and verify an interrupted output prefix')
    args = parser.parse_args()
    if args.positions <= 0:
        parser.error('Positive position count required')
    blocked = set()
    for name in ('puzzles_holdout', 'puzzles_final3000', 'stockfish_holdout', 'eval_holdout', 'endgame_holdout'):
        for line in Path('data', name + '.jsonl').read_text().splitlines():
            row = json.loads(line)
            board = chess.Board(row['fen'])
            blocked.add(' '.join(board.fen().split()[:4]))
            for move in row.get('moves', []):
                board.push_uci(move)
                blocked.add(' '.join(board.fen().split()[:4]))
    url = f'https://database.lichess.org/standard/lichess_db_standard_rated_{args.month}.pgn.zst'
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        parser.error('Use a new output path to preserve the existing corpus')
    temporary = out.with_suffix(out.suffix + '.part')
    if temporary.exists() and not args.resume:
        parser.error('An interrupted output exists; use --resume to verify and continue it')
    recovered = temporary.with_suffix('.recover') if temporary.exists() else temporary
    positions = games = skipped = 0
    with ExitStack() as stack:
        previous = stack.enter_context(temporary.open()) if recovered != temporary else None
        response = stack.enter_context(RangeReader(url))
        raw = stack.enter_context(zstandard.ZstdDecompressor().stream_reader(response))
        stream = stack.enter_context(io.TextIOWrapper(raw))
        saved = stack.enter_context(recovered.open('w'))
        while positions < args.positions:
            game = chess.pgn.read_game(stream, Visitor=Rated)
            if game is None:
                break
            if game.errors or not game.next() or game.headers.get('Result') not in ('1-0', '0-1', '1/2-1/2'):
                continue
            board = game.board()
            value = {'1-0': 1., '0-1': -1., '1/2-1/2': 0.}[game.headers['Result']]
            elo = min(int(game.headers[color + 'Elo']) for color in ('White', 'Black'))
            games += 1
            for move in game.mainline_moves():
                fen = board.fen()
                if ' '.join(fen.split()[:4]) in blocked:
                    skipped += 1
                else:
                    line = json.dumps({'fen': fen, 'move': move.uci(), 'value': value if board.turn else -value,
                        'game_id': game.headers['Site'], 'weight': 3. if elo >= 2600 else 2. if elo >= 2400 else 1.5 if elo >= 2200 else 1.}, separators=(',', ':')) + '\n'
                    if previous and (old := previous.readline()) and old != line:
                        raise ValueError('Interrupted corpus does not match the source archive')
                    saved.write(line)
                    positions += 1
                    if positions % 100000 == 0:
                        print(json.dumps({'positions': positions, 'games': games}), flush=True)
                    if positions == args.positions:
                        break
                board.push(move)
        if previous and previous.read(1):
            raise ValueError('Requested corpus is shorter than the existing prefix')
        source_headers = response.headers
    if positions != args.positions:
        raise ValueError('Archive ended before the requested position count')
    recovered.replace(out)
    temporary.unlink(missing_ok=True)
    out.with_suffix('.json').write_text(json.dumps({'url': url, 'source_headers': source_headers, 'license': 'CC0',
        'positions': positions, 'games': games, 'rating_range': [2000, 2800], 'time_controls': ['blitz', 'rapid'],
        'heldout_positions_skipped': skipped, 'history': 'Current-position corpus for the native 18-plane ResNet',
        'artifact': artifact_info(out)}, indent=2) + '\n')
    print(json.dumps({'positions': positions, 'games': games, 'artifact': artifact_info(out)}), flush=True)


if __name__ == '__main__':
    main()
