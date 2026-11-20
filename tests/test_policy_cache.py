import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from play.mlx import Graph
from train.mlx import losses
from train.policy_cache import cache_features, policy_views


def graph_fixture():
    rng = np.random.default_rng(27)
    graph = Graph.__new__(Graph)
    graph.weights = {name: mx.array(rng.normal(0, .2, shape).astype(np.float16)) for name, shape in
                     [('backbone', (4, 6)), ('/policy/w', (6, 5)), ('/value/w', (6, 3))]}
    graph.constants = {}
    graph.inputs, graph.outputs, graph.value_logits = ['input'], ['policy', 'wdl'], 'value'
    graph.nodes = [['MatMul', ['input', 'backbone'], ['hidden'], {}],
                   ['Relu', ['hidden'], ['features'], {}],
                   ['MatMul', ['features', '/policy/w'], ['policy'], {}],
                   ['MatMul', ['features', '/value/w'], ['value'], {}],
                   ['Softmax', ['value'], ['wdl'], {'axis': -1}]]
    graph.quantized = {}
    graph.run = mx.compile(graph.forward)
    return graph


def test_cached_head_matches_full_graph_logits_and_parameter_gradients():
    graph = graph_fixture()
    parameters = {'/policy/w': graph.weights['/policy/w'].astype(mx.float32)}
    frozen, head = policy_views(graph, parameters)
    inputs = mx.array(np.arange(12).reshape(3, 4) / 10, dtype=mx.float16)
    policy, features = frozen.run(inputs, graph.weights)
    mx.eval(policy, features)
    mask = mx.array([[1, 1, 0, 1, 1]] * 3, dtype=mx.bool_)
    target = mx.array([[1., 0, 0, 0, 0]] * 3)
    teacher = mx.softmax(mx.where(mask, policy.astype(mx.float32), -1e9), axis=-1)

    def objective(parameters, selected_graph, selected_input):
        return losses(selected_graph, {**graph.weights, **parameters}, selected_input, mask, target,
                      teacher, mx.zeros((3, 3)), value_weight=0, teacher_weight=.5, amp=True)[0]

    a, ga = mx.value_and_grad(objective)(parameters, graph, inputs)
    b, gb = mx.value_and_grad(objective)(parameters, head, features)
    np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-6)
    np.testing.assert_array_equal(np.array(ga['/policy/w']), np.array(gb['/policy/w']))
    expected = graph.forward(inputs, graph.weights)[0]
    actual = head.forward(features, graph.weights)[0]
    np.testing.assert_array_equal(np.array(actual), np.array(expected))
    assert len(head.nodes) == 1 and head.inputs == ['features']


def test_cache_reuse_and_changed_selection_or_artifact_rebuild(tmp_path):
    graph = graph_fixture()
    parameters = {'/policy/w': graph.weights['/policy/w']}
    class Dataset:
        def __len__(self):
            return 2
        def read_row(self, index):
            return {'index': index}
    calls = []
    def collate(rows):
        calls.append(rows)
        return mx.ones((len(rows), 4), dtype=mx.float16), None, None
    prefix = tmp_path / 'cached'
    _, first = cache_features(graph, parameters, Dataset(), prefix, 1, {'selection': 1}, collate)
    assert len(calls) == 2
    _, second = cache_features(graph, parameters, Dataset(), prefix, 1, {'selection': 1}, collate)
    assert len(calls) == 2
    np.testing.assert_array_equal(first[0], second[0])
    cache_features(graph, parameters, Dataset(), prefix, 1, {'selection': 2}, collate)
    assert len(calls) == 4
    corrupted = np.load(prefix.with_suffix('.features.npy'), mmap_mode='r+')
    corrupted[0, 0, 0] += 1
    corrupted.flush()
    cache_features(graph, parameters, Dataset(), prefix, 1, {'selection': 2}, collate)
    assert len(calls) == 6
    # Identical rows with a different frozen boundary must not reuse the old features.
    _, changed = cache_features(graph, {**parameters, 'backbone': graph.weights['backbone']},
                                Dataset(), prefix, 1, {'selection': 2}, collate)
    assert len(calls) == 8 and changed[0].shape == (2, 1, 4)


def test_cache_restores_flattened_square_features_with_variable_batch_sizes(tmp_path):
    graph = graph_fixture()
    graph.constants = {'flatten': np.array([-1, 4]), 'unflatten': np.array([-1, 10])}
    graph.nodes.insert(0, ['Reshape', ['input', 'flatten'], ['flat'], {}])
    graph.nodes[1][1][0] = 'flat'
    graph.nodes[3][2] = ['square_policy']
    graph.nodes.insert(4, ['Reshape', ['square_policy', 'unflatten'], ['policy'], {}])
    graph.run = mx.compile(graph.forward)
    class Dataset:
        def __len__(self):
            return 3
        def read_row(self, index):
            return {'index': index}
    def collate(rows):
        return mx.array([[[row['index'] + 1.] * 4] * 2 for row in rows], dtype=mx.float16), None, None
    head, arrays = cache_features(graph, {'/policy/w': graph.weights['/policy/w']}, Dataset(),
                                 tmp_path / 'cached', 2, {'selection': 1}, collate)
    assert arrays[0].shape == (3, 2, 6) and arrays[1].shape == (3, 10)
    for selected in ([2, 0], [1]):
        inputs, _, _ = collate([Dataset().read_row(i) for i in selected])
        features = mx.array(arrays[0][selected].reshape(-1, *arrays[0].shape[2:]))
        expected = graph.forward(inputs, graph.weights)[0]
        actual = head.forward(features, graph.weights)[0]
        np.testing.assert_array_equal(np.array(actual), np.array(expected))
