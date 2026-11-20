import chess
import pytest

from eval.arena import play_match, present_outcome


def cycle(plies):
    board = chess.Board()
    for move in (['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2)[:plies]:
        board.push_uci(move)
    return board


def test_an_available_repetition_claim_does_not_force_the_engine_to_draw():
    board = cycle(7)
    assert board.can_claim_threefold_repetition() and not board.is_repetition(3)
    assert present_outcome(board) is None
    board.push_uci('f6e4')
    assert present_outcome(board) is None
    for move in ['g2g4', 'e7e5', 'f2f3', 'd8h4']:
        board.push_uci(move)
    assert present_outcome(board).result() == '0-1'


def test_a_played_third_occurrence_is_a_draw():
    assert present_outcome(cycle(8)) == chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)


def test_fifty_move_claim_is_available_only_after_the_nonzeroing_move():
    board = chess.Board('7k/8/8/8/8/8/P7/R5K1 w - - 99 51')
    assert board.can_claim_fifty_moves() and not board.is_fifty_moves()
    assert present_outcome(board) is None
    pawn = board.copy();pawn.push_uci('a2a3')
    assert present_outcome(pawn) is None
    board.push_uci('a1b1')
    assert present_outcome(board) == chess.Outcome(chess.Termination.FIFTY_MOVES, None)


@pytest.mark.parametrize('avoid, reason, plies', [(True, 'checkmate', 24), (False, 'threefold_repetition', 16)])
def test_full_paired_games_can_avoid_a_prospective_draw_or_play_the_repetition(tmp_path, avoid, reason, plies):
    moves = ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1']
    moves += ['f6e4', 'g2g4', 'e7e5', 'f2f3', 'd8h4'] if avoid else ['f6g8']
    class Runner:
        forward_calls = 0
        def select_move(self, board, deterministic):
            self.forward_calls += 1
            return board.parse_uci(moves[len(board.move_stack)])
    a, b = Runner(), Runner()
    result = play_match(a, b, num_games=2, max_plies=20, pgn_path=tmp_path / 'games.pgn')
    assert result['terminations'] == {reason: 2}
    assert result['moves_a'] + result['moves_b'] == plies
    assert result['neural_forward_calls_a'] == result['moves_a']
    assert result['neural_forward_calls_b'] == result['moves_b']
    assert result['draw_claim_protocol'] == 'present_position'
