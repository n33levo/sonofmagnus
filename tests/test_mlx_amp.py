import numpy as np
import pytest


def test_scaled_half_precision_keeps_masked_losses_and_saturated_wdl_gradients_finite():
    mx = pytest.importorskip('mlx.core')
    from train.mlx import losses

    class Graph:
        def forward(self, inputs, weights, raw_value=False):
            return inputs @ weights['policy'], inputs @ weights['value']

    inputs = mx.eye(2).astype(mx.float16)
    weights = {'policy': mx.array([[.3, -.2, .1, .5], [.2, -.4, .7, -.1]]),
               'value': mx.array([[50., -50., -50.], [-50., 50., -50.]])}
    mask = mx.array([[True, False, True, False], [False, True, False, True]])
    target = mx.array([[1., 0., 0., 0.], [0., 0., 0., 1.]])
    teacher = mx.array([[0., 1., 0.], [1., 0., 0.]])

    def objective(parameters, amp=False):
        values = losses(Graph(), parameters, inputs, mask, target, target, teacher,
                        sample_weight=mx.array([3., 1.]), amp=amp)
        return (values[0] * (1024 if amp else 1), *values[1:])

    reference, expected = mx.value_and_grad(objective)(weights)
    actual, scaled = mx.value_and_grad(lambda p: objective(p, True))(weights)
    assert np.isfinite(float(actual[0]))
    assert float(actual[0]) / 1024 == pytest.approx(float(reference[0]), abs=.005)
    for name, gradient in scaled.items():
        assert gradient.dtype == mx.float32
        np.testing.assert_allclose(np.array(gradient) / 1024, np.array(expected[name]), atol=.002, rtol=.01)
    assert float(mx.abs(scaled['value']).max()) > 0
