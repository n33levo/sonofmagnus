import hashlib
import json
import random

import chess
import chess.pgn
import zstandard

from chessio.records import record_board
from scripts.gm_corpus import build, eligible, eval_drop


def test_gm_filter_requires_two_verified_titles_and_ratings():
    headers = {'WhiteTitle': 'GM', 'BlackTitle': 'GM', 'WhiteElo': '2750',
               'BlackElo': '2500', 'Result': '1-0'}
    assert eligible(headers, 2500)
    for key, value in [('BlackTitle', 'BOT'), ('WhiteTitle', 'IM'), ('BlackElo', '2499'),
                       ('WhiteElo', '?'), ('Result', '*'), ('Variant', 'Chess960')]:
        assert not eligible(headers | {key: value}, 2500)


def test_recorded_eval_drop_uses_the_mover_colour():
    game = chess.pgn.Game()
    node = game.add_variation(chess.Move.from_uci('e2e4'))
    game.comment, node.comment = '[%eval 0.20]', '[%eval 1.20]'
    assert eval_drop(game, node, chess.BLACK) == 100
    assert eval_drop(game, node, chess.WHITE) == 0
    node.comment = ''
    assert eval_drop(game, node, chess.BLACK) is None


def test_corpus_replays_and_keeps_games_and_positions_disjoint(tmp_path):
    games = []
    rng = random.Random(4)
    for number in range(64):
        game, board = chess.pgn.Game(), chess.Board()
        game.headers.update(White=f'GM {number}', Black='Other GM', WhiteTitle='GM',
                            BlackTitle='GM', WhiteElo='2750', BlackElo='2650', Result='1/2-1/2')
        node = game
        for _ in range(24):
            if board.is_game_over():
                break
            move = rng.choice(list(board.legal_moves))
            node = node.add_variation(move)
            board.push(move)
        games.append(game)
    archive = tmp_path / 'gm.pgn.zst'
    archive.write_bytes(zstandard.ZstdCompressor().compress(
        '\n\n'.join(map(str, games + [games[0]])).encode()))
    blocked = games[0].mainline().__iter__()
    for _ in range(17):
        position = next(blocked).board()
    excluded = tmp_path / 'test.jsonl'
    excluded.write_text(json.dumps({'fen': position.fen()}) + '\n')
    train, validation = tmp_path / 'train.jsonl', tmp_path / 'val.jsonl'
    report = build([archive], train, validation, [excluded], max_positions=10000)
    splits = [[json.loads(line) for line in path.open()] for path in (train, validation)]
    assert all(splits)
    assert not {r['game_id'] for r in splits[0]} & {r['game_id'] for r in splits[1]}
    keys = [{' '.join(r['fen'].split()[:4]) for r in rows} for rows in splits]
    assert not keys[0] & keys[1]
    assert ' '.join(position.fen().split()[:4]) not in keys[0] | keys[1]
    assert report['counts']['positions'] == sum(map(len, splits))
    assert report['counts']['games'] <= 64
    for row in splits[0] + splits[1]:
        board = record_board(row)
        assert chess.Move.from_uci(row['move']) in board.legal_moves
    assert report['training']['sha256'] == hashlib.sha256(train.read_bytes()).hexdigest()
