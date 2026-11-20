import json

import chess
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
pytest.importorskip('onnxruntime')

from chessio.lc0 import encode_history
from deploy.mlx_to_onnx import export
from model.artifacts import artifact_info
from play.mlx import Graph
from play.onnx import OnnxRunner


@pytest.mark.parametrize('quantized', [False, True])
def test_cpu_export_preserves_logits_wdl_and_source_with_variable_batches(tmp_path, quantized):
    import onnxruntime as ort
    source = tmp_path / 'source'
    source.mkdir()
    rng = np.random.default_rng(7)
    weights = {'shape': mx.array([-1, 7168]),
        'features': mx.array(rng.normal(0, .01, (7168, 56)).astype(np.float16)),
        'middle_w': mx.array(rng.normal(0, .01, (56, 64)).astype(np.float16)),
        'policy_w': mx.array(rng.normal(0, .01, (64, 1858)).astype(np.float16)),
        'value_w': mx.array(rng.normal(0, .01, (64, 3)).astype(np.float16))}
    nodes = [['Reshape', ['planes', 'shape'], ['flat'], {}],
        ['MatMul', ['flat', 'features'], ['hidden'], {}],
        ['Relu', ['hidden'], ['activated'], {}],
        ['MatMul', ['activated', 'middle_w'], ['middle'], {}],
        ['MatMul', ['middle', 'policy_w'], ['scores'], {}],
        ['MatMul', ['middle', 'value_w'], ['wdl_logits'], {}],
        ['Softmax', ['wdl_logits'], ['wdl'], {'axis': -1}]]
    configurations = {}
    if quantized:
        from deploy.export_mlx import quantize_weights
        configurations = quantize_weights(weights, {'features', 'middle_w', 'policy_w', 'value_w'}, 8)
        assert configurations['features'][2] and not configurations['middle_w'][2]
    (source / 'graph.json').write_text(json.dumps({'inputs': ['planes'], 'outputs': ['scores', 'wdl'],
        'integer_constants': ['shape'], 'nodes': nodes, 'quantized': configurations}))
    (source / 'metadata.json').write_text('{}')
    mx.save_safetensors(str(source / 'weights.safetensors'), weights)
    frozen, output = artifact_info(source), tmp_path / 'model.onnx'
    export(source, output)
    assert artifact_info(source) == frozen
    graph, runner = Graph(source), OnnxRunner(output)
    parameters = {name: value.astype(mx.float32) for name, value in graph.weights.items()}
    for name, (group, bits, transpose) in graph.quantized.items():
        decoded = mx.dequantize(graph.weights[name + '.packed'], graph.weights[name + '.scales'].astype(mx.float32),
            graph.weights[name + '.biases'].astype(mx.float32), group_size=group, bits=bits)
        parameters[name] = decoded.T if transpose else decoded
    graph.quantized = {}
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    boards = [chess.Board(), chess.Board('7k/8/8/3pP3/8/8/8/K7 w - d6 0 1')]
    for selected in (boards[:1], boards):
        inputs = np.stack([encode_history(board) for board in selected])
        reference = graph.forward(mx.array(inputs), parameters)
        actual = session.run(None, {'planes': inputs})
        for expected, value in zip(reference, actual):
            np.testing.assert_allclose(value, np.array(expected), atol=2e-6, rtol=2e-5)
        for board in selected:
            assert runner.select_move(board, deterministic=True) in board.legal_moves
    with pytest.raises(ValueError, match='new output'):
        export(source, output)
