import numpy as np
import pytest


def test_half_gradient_overflow_retries_the_same_batch_without_mutating_parameters():
    mx = pytest.importorskip('mlx.core')
    from train.mlx import losses, finite_gradients
    class Graph:
        def forward(self, inputs, weights, raw_value=False):
            return inputs @ weights['policy'], inputs @ weights['value']
    weights = {'policy': mx.zeros((2, 2)), 'value': mx.zeros((2, 3))}
    before = {name: np.array(value) for name, value in weights.items()}
    inputs, mask, target = mx.array([[200., 200.]], dtype=mx.float16), mx.array([[True, True]]), mx.array([[1., 0.]])
    def objective(parameters, scale):
        values = losses(Graph(), parameters, inputs, mask, target, target, mx.array([[0., 1., 0.]]),
                        teacher_weight=0., value_weight=0., amp=True)
        return (values[0] * scale, *values[1:])
    values, gradients, norm, scale, retries = finite_gradients(mx.compile(mx.value_and_grad(objective)), weights, (), 1024, True)
    assert scale == 512 and retries == 1 and float(norm) == pytest.approx(200.)
    assert float(values[0]) == pytest.approx(np.log(2), abs=1e-5)
    np.testing.assert_allclose(np.array(gradients['policy']), [[-100., 100.], [-100., 100.]])
    for name, value in weights.items():
        np.testing.assert_array_equal(np.array(value), before[name])


def test_student_history_dropout_matches_fen_encoding_without_changing_teacher_input():
    mx = pytest.importorskip('mlx.core')
    import chess
    from chessio.lc0 import encode_history
    from train.mlx import batch_rows, drop_history
    histories = [['e2e4', 'e7e5', 'g1f3'], ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2,
                 ['e2e4', 'a7a6', 'e4e5', 'd7d5']]
    rows = []
    for history in histories:
        board = chess.Board()
        for move in history:
            board.push_uci(move)
        rows.append({'fen': board.fen(), 'root_fen': chess.STARTING_FEN, 'history': history,
                     'move': next(iter(board.legal_moves)).uci()})
    original, mask, target = batch_rows(rows)
    original = original.astype(mx.float16)
    before = np.array(original)
    actual = np.array(drop_history(original, rows, [True, False, True]))
    np.testing.assert_array_equal(np.array(original), before)
    np.testing.assert_array_equal(actual[1], before[1])
    for index in (0, 2):
        np.testing.assert_array_equal(actual[index], encode_history(chess.Board(rows[index]['fen'])))
    assert not np.array_equal(actual[0], before[0])
    assert np.array(mask).sum() > 0 and np.array(target).sum() == 3


def test_mlx_convolution_layout_groups_bias_and_pooling_match_torch(tmp_path):
    mx = pytest.importorskip('mlx.core')
    import json
    import torch
    from play.mlx import Graph
    rng = np.random.default_rng(0)
    weights = {'kernel': rng.normal(size=(6, 2, 3, 3)).astype(np.float32),
               'bias': rng.normal(size=6).astype(np.float32),
               'policy': rng.normal(size=(6, 5)).astype(np.float32),
               'value': rng.normal(size=(6, 3)).astype(np.float32)}
    nodes = [['Conv', ['x', 'kernel', 'bias'], ['conv'], {'pads': [1, 1, 1, 1], 'strides': [2, 1], 'group': 2}],
             ['Relu', ['conv'], ['activated'], {}],
             ['Split', ['activated'], ['first', 'second'], {'axis': 1, 'split': [3, 3]}],
             ['Concat', ['first', 'second'], ['rejoined'], {'axis': 1}],
             ['ReduceMean', ['rejoined'], ['pooled'], {'axes': [2, 3], 'keepdims': 0}],
             ['MatMul', ['pooled', 'policy'], ['policy_out'], {}],
             ['MatMul', ['pooled', 'value'], ['value_logits'], {}],
             ['Softmax', ['value_logits'], ['wdl'], {'axis': -1}]]
    (tmp_path / 'graph.json').write_text(json.dumps({'inputs': ['x'], 'outputs': ['policy_out', 'wdl'],
                                                   'integer_constants': [], 'nodes': nodes}))
    mx.save_safetensors(str(tmp_path / 'weights.safetensors'), {k: mx.array(v) for k, v in weights.items()})
    graph = Graph(tmp_path)
    inputs = rng.normal(size=(2, 4, 5, 5)).astype(np.float32)
    conv = torch.nn.functional.conv2d(torch.from_numpy(inputs), torch.from_numpy(weights['kernel']),
                                     torch.from_numpy(weights['bias']), stride=(2, 1), padding=1, groups=2)
    features = conv.relu().mean((2, 3))
    expected = (features @ torch.from_numpy(weights['policy']),
                (features @ torch.from_numpy(weights['value'])).softmax(-1))
    actual = graph.run(mx.array(inputs), graph.weights)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(np.array(a), b.numpy(), atol=2e-5, rtol=2e-5)
    gradient = mx.grad(lambda kernel: graph.forward(mx.array(inputs), {**graph.weights, 'kernel': kernel})[0].sum())(graph.weights['kernel'])
    assert np.isfinite(np.array(gradient)).all() and float(mx.abs(gradient).sum()) > 0


def test_cnn_widening_preserves_se_paths_and_activates_added_head_weights(tmp_path):
    mx = pytest.importorskip('mlx.core')
    import json
    from deploy.widen_mlx import widen
    from play.mlx import Graph
    source, output = tmp_path / 'source', tmp_path / 'wide'
    source.mkdir()
    rng = np.random.default_rng(0)
    shapes = {'/conv/w': (128, 3, 1, 1), '/conv/b': (128,), '/block/se/first': (128, 32),
              '/block/se/second': (32, 256), '/block/se/bias': (256,), '/policy/w': (128, 5), '/value/w': (128, 3)}
    weights = {name: mx.array(rng.normal(size=shape).astype(np.float32)) for name, shape in shapes.items()}
    weights['shape'] = mx.array([-1, 256, 1, 1])
    nodes = [['Conv', ['x', '/conv/w', '/conv/b'], ['conv'], {}],
             ['ReduceMean', ['conv'], ['mean'], {'axes': [2, 3], 'keepdims': 0}],
             ['MatMul', ['mean', '/block/se/first'], ['hidden'], {}],
             ['Relu', ['hidden'], ['relu'], {}],
             ['MatMul', ['relu', '/block/se/second'], ['gate'], {}],
             ['Add', ['gate', '/block/se/bias'], ['biased'], {}],
             ['Reshape', ['biased', 'shape'], ['shaped'], {}],
             ['Split', ['shaped'], ['scale', 'offset'], {'axis': 1}],
             ['Sigmoid', ['scale'], ['sigmoid'], {}],
             ['Mul', ['conv', 'sigmoid'], ['scaled'], {}],
             ['Add', ['scaled', 'offset'], ['features'], {}],
             ['ReduceMean', ['features'], ['pooled'], {'axes': [2, 3], 'keepdims': 0}],
             ['MatMul', ['pooled', '/policy/w'], ['policy'], {}],
             ['MatMul', ['pooled', '/value/w'], ['value'], {}],
             ['Softmax', ['value'], ['wdl'], {'axis': -1}]]
    (source / 'graph.json').write_text(json.dumps({'inputs': ['x'], 'outputs': ['policy', 'wdl'],
                                                 'integer_constants': ['shape'], 'nodes': nodes}))
    (source / 'metadata.json').write_text('{}')
    mx.save_safetensors(str(source / 'weights.safetensors'), weights)
    metadata = widen(source, output, channels=160, seed=0)
    assert metadata['widened_channels'] == 160
    base, wide = Graph(source), Graph(output)
    inputs = mx.array(rng.normal(size=(2, 3, 5, 5)).astype(np.float32))
    for actual, expected in zip(wide.run(inputs, wide.weights), base.run(inputs, base.weights)):
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=.001, rtol=2e-5)
    gradient = mx.grad(lambda head: wide.forward(inputs, {**wide.weights, '/policy/w': head})[0].sum())(wide.weights['/policy/w'])
    assert float(mx.abs(gradient[128:]).sum()) > 0


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
