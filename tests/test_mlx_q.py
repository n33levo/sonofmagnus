import chess
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from chessio.lc0 import policy_index
from train.mlx import batch_rows, losses


def test_q_targets_keep_unknown_actions_unlabelled():
    row = {'fen': chess.STARTING_FEN, 'move': 'e2e4', 'q_values': {'e2e4': .7, 'd2d4': .6}}
    _, legal, _, target, mask = batch_rows([row], q_values=True)
    expected = [policy_index(chess.Move.from_uci(move), chess.Board()) for move in ('e2e4', 'd2d4')]
    assert int(mask.sum()) == 2 and int(legal.sum()) == 20
    np.testing.assert_allclose(np.array(target)[0, expected], [.7, .6])
    assert not bool(mask[0, policy_index(chess.Move.from_uci('g1f3'), chess.Board())])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.1, 1.1])
def test_q_targets_reject_invalid_probabilities(value):
    with pytest.raises(ValueError, match='Q labels'):
        batch_rows([{'fen': chess.STARTING_FEN, 'move': 'e2e4', 'q_values': {'e2e4': value}}], q_values=True)


def test_q_targets_reject_illegal_actions():
    with pytest.raises(ValueError, match='Q labels'):
        batch_rows([{'fen': chess.STARTING_FEN, 'move': 'e2e4', 'q_values': {'e2e5': .7}}], q_values=True)


def test_q_gradients_ignore_unknown_actions_and_rows_without_labels():
    class Graph:
        def forward(self, inputs, weights, raw_value=False):
            return weights['policy'], mx.zeros((2, 3))
    parameters = {'policy': mx.array([[0., 100., -100.], [10., -10., 100.]])}
    mask = mx.ones((2, 3), dtype=mx.bool_)
    target = mx.array([[.8, 0., 0.], [0., 0., 0.]])
    known = mx.array([[True, False, False], [False, False, False]])
    def objective(weights):
        return losses(Graph(), weights, mx.zeros((2, 1)), mask, mx.zeros((2, 3)),
                      mx.zeros((2, 3)), mx.zeros((2, 3)), value_weight=0, teacher_weight=0,
                      q_target=target, q_mask=known, q_value_weight=1)[-1]
    value, gradient = mx.value_and_grad(objective)(parameters)
    assert float(value) == pytest.approx(np.log(2))
    np.testing.assert_allclose(np.array(gradient['policy']), [[-.3, 0., 0.], [0., 0., 0.]], atol=1e-7)
    empty = losses(Graph(), parameters, mx.zeros((2, 1)), mask, mx.zeros((2, 3)),
                   mx.zeros((2, 3)), mx.zeros((2, 3)), value_weight=0,
                   q_target=target, q_mask=mx.zeros((2, 3), dtype=mx.bool_), q_value_weight=1)
    assert float(empty[-1]) == 0 and np.isfinite(float(empty[0]))


def test_combined_objective_uses_relative_q_ranking_when_requested():
    class Graph:
        def forward(self, inputs, weights, raw_value=False):
            return weights['policy'], mx.zeros((1, 3))

    def objective(policy, kind):
        return losses(Graph(), {'policy': policy}, mx.zeros((1, 1)),
            mx.ones((1, 3), dtype=mx.bool_), mx.array([[1., 0., 0.]]),
            mx.zeros((1, 3)), mx.zeros((1, 3)), value_weight=0, teacher_weight=0,
            q_target=mx.array([[.9, .1, 0.]]), q_mask=mx.array([[True, True, False]]),
            q_value_weight=1, q_objective=kind)[-1]

    policy = mx.array([[0., 1., 100.]])
    assert float(objective(policy, 'ranking')) == pytest.approx(np.logaddexp(0, 1))
    assert float(objective(policy + 30, 'ranking')) == pytest.approx(float(objective(policy, 'ranking')))
    assert float(objective(policy, 'calibration')) != pytest.approx(float(objective(policy + 30, 'calibration')))
