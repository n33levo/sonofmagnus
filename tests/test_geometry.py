import json

import chess
import numpy as np
import pytest

from deploy.add_geometry import piece_geometry, add_geometry


def test_geometry_matches_empty_board_attacks_for_both_orientations():
    basis = piece_geometry()
    for side, color in enumerate((chess.WHITE, chess.BLACK)):
        for piece in chess.PIECE_TYPES:
            for square in chess.SQUARES:
                board = chess.Board(None)
                board.set_piece_at(square, chess.Piece(piece, color))
                expected = np.array([target in board.attacks(square) for target in chess.SQUARES])
                np.testing.assert_array_equal(basis[side * 6 + piece - 1, square], expected)


def test_geometry_preserves_initial_policy_and_has_trainable_new_paths(tmp_path):
    mx = pytest.importorskip('mlx.core')
    import onnxruntime as ort
    from play.mlx import Graph
    from deploy.export_attention import export
    from chessio.lc0 import encode_history
    rng = np.random.default_rng(20261011)
    source, augmented = tmp_path / 'source.mlx', tmp_path / 'augmented.mlx'
    source.mkdir()
    arrays = {'kernel': rng.normal(0, .002, (4, 112, 3, 3)).astype(np.float32),
              'policy_w': rng.normal(0, .01, (4, 1858)).astype(np.float32),
              'value_w': rng.normal(0, .01, (4, 3)).astype(np.float32)}
    nodes = [['Conv', ['board', 'kernel'], ['conv'], {'pads': [1, 1, 1, 1]}],
             ['ReduceMean', ['conv'], ['pooled'], {'axes': [2, 3], 'keepdims': 0}],
             ['MatMul', ['pooled', 'policy_w'], ['policy'], {}],
             ['MatMul', ['pooled', 'value_w'], ['value_logits'], {}],
             ['Softmax', ['value_logits'], ['wdl'], {'axis': -1}]]
    (source / 'graph.json').write_text(json.dumps({'inputs': ['board'], 'outputs': ['policy', 'wdl'], 'integer_constants': [], 'nodes': nodes}))
    (source / 'metadata.json').write_text('{}')
    mx.save_safetensors(str(source / 'weights.safetensors'), {k: mx.array(v) for k, v in arrays.items()})
    add_geometry(source, augmented)
    old, new = Graph(source), Graph(augmented)
    board = chess.Board()
    board.push_uci('e2e4')
    inputs = np.stack([encode_history(chess.Board()), encode_history(board)])
    reference = old.run(mx.array(inputs), old.weights)
    actual = new.run(mx.array(inputs), new.weights)
    for a, b in zip(actual, reference):
        np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-6, rtol=1e-5)
    gradient = mx.grad(lambda kernel: new.forward(mx.array(inputs), {**new.weights, 'kernel': kernel})[0].sum())(new.weights['kernel'])
    assert float(mx.abs(gradient[:, 112:]).sum()) > 0
    metadata = json.loads((augmented / 'metadata.json').read_text())
    assert metadata['frozen_weights'] == ['/const/geometry/basis']
    assert metadata['trainable_scalars'] == 11908
    assert metadata['stored_scalars_before_quantization'] == 11908 + 49152
    exported = tmp_path / 'geometry.onnx'
    export(augmented, exported)
    cpu = ort.InferenceSession(str(exported), providers=['CPUExecutionProvider']).run(None, {'board': inputs})
    for a, b in zip(cpu, reference):
        np.testing.assert_allclose(a, np.array(b), atol=1e-6, rtol=1e-5)
    import onnx
    from deploy.quantize_int8 import quantize_model
    calibration = tmp_path / 'calibration.jsonl'
    calibration.write_text(json.dumps({'fen': chess.STARTING_FEN}) + '\n')
    mixed = tmp_path / 'mixed.onnx'
    quantize_model(exported, mixed, calibration_data=calibration, keep_input_and_heads=True)
    initializers = onnx.load(mixed).graph.initializer
    for shape in ([4, 124, 3, 3], [12, 64, 64]):
        assert any(list(weight.dims) == shape and weight.data_type == onnx.TensorProto.FLOAT for weight in initializers)
