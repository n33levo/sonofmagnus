import chess
import pytest

from scripts.build_q_corpus import complete_label


@pytest.mark.parametrize('black', [False, True])
def test_full_action_labels_use_relative_scores_and_all_legal_moves(black):
    board = chess.Board()
    if black:
        board.push_uci('e2e4')
    moves = list(board.legal_moves)
    values = [{'line': move.uci(), 'cp': (300 if i == 0 else -300) * (-1 if black else 1)} for i, move in enumerate(moves)]
    label = complete_label({'fen': board.fen(), 'evals': [{'depth': 14, 'pvs': values}]})
    assert set(label['q_values']) == {move.uci() for move in moves}
    assert max(label['move_probs'], key=label['move_probs'].get) == moves[0].uci()
    assert label['q_values'][moves[0].uci()] > .5
    assert all(q < .5 for move, q in label['q_values'].items() if move != moves[0].uci())
    assert sum(label['move_probs'].values()) == pytest.approx(1)


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'illegal', 'shallow', 'nonfinite'])
def test_partial_or_invalid_action_evaluations_are_rejected(failure):
    board = chess.Board()
    values = [{'line': move.uci(), 'cp': 0} for move in board.legal_moves]
    depth = 14
    if failure == 'missing':
        values.pop()
    elif failure == 'duplicate':
        values[-1] = values[0]
    elif failure == 'illegal':
        values[-1]['line'] = 'a1a8'
    elif failure == 'shallow':
        depth = 11
    else:
        values[-1]['cp'] = float('nan')
    assert complete_label({'fen': board.fen(), 'evals': [{'depth': depth, 'pvs': values}]}) is None
