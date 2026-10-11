import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from train.mlx import losses


def test_expert_values_use_only_real_labels_and_ignore_absent_values():
    class Graph:
        def forward(self, inputs, weights, raw_value=False):
            return mx.zeros((2, 2)), weights['value']

    weights = {'value': mx.array([[.2, .1, -.3], [4., -2., -3.]])}
    policy = mx.array([[1., 0.], [1., 0.]])
    mask = mx.ones((2, 2), dtype=mx.bool_)
    args = (Graph(), weights, mx.zeros((2, 1)), mask, policy, policy, mx.zeros((2, 3)))
    first = losses(*args, value_target=mx.array([.6, -1.]), value_mask=mx.array([1., 0.]))
    second = losses(*args, value_target=mx.array([.6, 1.]), value_mask=mx.array([1., 0.]))
    probabilities = np.exp(np.array(weights['value'])[0])
    probabilities /= probabilities.sum()
    expected = (probabilities[0] - probabilities[2] - .6) ** 2
    assert float(first[-1]) == pytest.approx(expected, rel=1e-6)
    assert float(first[-1]) == float(second[-1])
    empty = losses(*args, value_target=mx.array([1., -1.]), value_mask=mx.zeros((2,)))
    assert float(empty[-1]) == 0 and np.isfinite(float(empty[0]))
