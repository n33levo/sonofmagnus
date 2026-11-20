import chess

from chessio.records import record_board
from scripts.collect_history import game_positions


def game(game_id, weight=2):
    board = chess.Board()
    rows = []
    for san in 'e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7'.split():
        move = board.parse_san(san).uci()
        rows.append({'fen': board.fen(), 'move': move, 'game_id': game_id, 'weight': weight})
        board.push_uci(move)
    return rows


def test_history_selection_replays_real_moves_once_per_game_and_skips_gaps():
    rows = game('complete') + game('gap')[:5] + game('gap')[6:] + game('too-low', 1.5) + game('second')
    selected = list(game_positions(rows, set(), seed=3))
    assert {r['game_id'] for r in selected} == {'complete', 'second'}
    for row in selected:
        assert len(row['history']) >= 16
        board = record_board(row)
        assert board.fen() == row['fen'] and chess.Move.from_uci(row['move']) in board.legal_moves


def test_history_selection_does_not_invent_missing_early_moves_or_use_frozen_positions():
    rows = game('truncated')[1:] + game('blocked')
    blocked = {' '.join(r['fen'].split()[:4]) for r in rows}
    assert list(game_positions(rows, blocked)) == []


def test_explicit_null_history_is_a_fen_only_record():
    fen = chess.Board().fen()
    assert record_board({'fen': fen, 'root_fen': None, 'history': None}).fen() == fen
