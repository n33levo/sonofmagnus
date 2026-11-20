import chess
import numpy as np
import pytest

from chessio.policy_map import move_to_policy_index
from eval.openings import evaluate


def test_opening_agreement_batches_real_histories_and_scores_each_ply():
    class Runner:
        forward_calls = 0
        histories = []

        def batch_policy(self, boards):
            self.forward_calls += 1
            result = []
            for board in boards:
                self.histories.append(len(board.move_stack))
                move = chess.Move.from_uci(('e2e4', 'c7c5', 'b1c3', 'e7e5')[len(board.move_stack)])
                assert move in board.legal_moves
                logits = np.zeros((8, 8, 73), np.float32)
                square, plane = move_to_policy_index(move, board)
                logits[square // 8, square % 8, plane] = 10
                result.append((logits, 0.))
            return result

    runner = Runner()
    row = {'opening': 'Sicilian Defense', 'moves': ['e2e4', 'c7c5', 'g1f3', 'd7d6']}
    result = evaluate(runner, [row], batch_size=3, depths={'Sicilian Defense': 2})
    assert runner.histories == [0, 1, 2, 3] and runner.forward_calls == 2
    assert result['positions_evaluated'] == 4
    assert result['openings']['Sicilian Defense']['top1_agreement'] == .5
    assert result['openings']['Sicilian Defense']['agreement_by_fullmove'] == [1., 0.]
    with pytest.raises(ValueError, match='complete evaluation prefix'):
        evaluate(runner, [row])
