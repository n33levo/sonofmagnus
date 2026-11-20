import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from train.q_rank import ranking_loss


def test_ranking_pushes_winner_up_loser_down_and_ignores_unknown():
    target = mx.array([[.9, .1, 0.]])
    known = mx.array([[True, True, False]])
    policy = mx.array([[0., 1., float('inf')]])
    value, grad = mx.value_and_grad(lambda p: ranking_loss(p, target, known))(policy)
    assert float(value) == pytest.approx(np.logaddexp(0, 1))
    assert float(grad[0, 0]) < 0 < float(grad[0, 1])
    assert float(grad[0, 2]) == 0
    assert float(grad.sum()) == pytest.approx(0, abs=1e-7)
    assert float(ranking_loss(policy + 50, target, known)) == pytest.approx(float(value))


def test_equal_values_and_unlabelled_rows_contribute_no_gradient():
    target = mx.array([[.5, .5, 0.], [.9, .1, 0.]])
    known = mx.array([[True, True, False], [False, False, False]])
    value, grad = mx.value_and_grad(lambda p: ranking_loss(p, target, known))(mx.ones((2, 3)))
    assert float(value) == 0
    np.testing.assert_array_equal(np.array(grad), np.zeros((2, 3)))


def test_tied_best_actions_share_credit_and_weights_exclude_empty_rows():
    target = mx.array([[.9, .9, .1], [.5, .5, .5]])
    policy = mx.array([[0., 0., 1.], [100., -100., 0.]])
    known = mx.ones((2, 3), dtype=mx.bool_)
    value, grad = mx.value_and_grad(lambda p: ranking_loss(p, target, known, mx.array([3., 100.])))(policy)
    assert float(value) == pytest.approx(np.logaddexp(0, 1))
    assert float(grad[0, 0]) == pytest.approx(float(grad[0, 1]))
    np.testing.assert_array_equal(np.array(grad[1]), np.zeros(3))
