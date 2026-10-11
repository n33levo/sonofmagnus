import hashlib
import json
import random
from pathlib import Path
from unittest.mock import patch

import chess
import numpy as np
import pytest

from chessio.lc0 import encode_history, policy_index
from eval.arena import UciRunner, play_match
from eval.rating import fit_rating, summarize_pool
from model.artifacts import artifact_info


def test_rating_recovers_known_expected_scores_and_boundaries():
    anchors = np.array([2512., 2714., 2933.])
    scores = 1 / (1 + 10 ** ((anchors - 2650) / 400))
    assert fit_rating(scores, anchors) == pytest.approx(2650)
    assert np.isneginf(fit_rating(np.zeros(3), anchors))
    assert np.isposinf(fit_rating(np.ones(3), anchors))
    with pytest.raises(ValueError):
        fit_rating([.5, float('nan'), 2], anchors)


def test_rating_clusters_shared_openings_and_reports_unbounded_results():
    scores = np.array([[[1., 0.], [.5, .5]]] * 3)
    result = summarize_pool(scores, [2500, 2600, 2700], samples=100)
    assert result['local_elo'] == pytest.approx(2600)
    assert result['independent_opening_clusters'] == 2
    assert result['games'] == 12
    assert result['elo_opening_bootstrap_95'] == pytest.approx([2600, 2600])
    assert result['elo_hoeffding_95'] == [None, None]
    all_wins = summarize_pool(np.ones((1, 10, 2)), [2500], samples=100)
    assert all_wins['local_elo'] is None
    assert all_wins['elo_opening_bootstrap_95'] == [None, None]
    with pytest.raises(ValueError):
        summarize_pool(np.full((1, 2, 2), .75), [2500])


def test_artifact_size_and_digest_include_entire_directory(tmp_path):
    directory = tmp_path / 'model.mlx'
    directory.mkdir()
    (directory / 'weights').write_bytes(b'123')
    (directory / 'graph').write_bytes(b'4567')
    before = artifact_info(directory)
    assert before['model_bytes'] == 7
    assert artifact_info(directory) == before
    (directory / 'graph').write_bytes(b'4568')
    assert artifact_info(directory)['sha256'] != before['sha256']
    assert artifact_info(directory / 'weights')['sha256'] == hashlib.sha256(b'123').hexdigest()


def test_uci_opponent_receives_actual_clocks_and_new_game():
    with patch('chess.engine.SimpleEngine.popen_uci') as open_engine:
        open_engine.return_value.options = {'Threads': None, 'Hash': None}
        runner = UciRunner('/test/engine')
        token = runner.game
        runner.new_game()
        assert runner.game != token
        runner.set_clock(12., 7., .3)
        runner.select_move(chess.Board())
        call = open_engine.return_value.play.call_args
        assert call.args[1].white_clock == 12.
        assert call.args[1].black_clock == 7.
        assert call.args[1].white_inc == .3
        assert call.args[1].time is None
        assert 'UCI_LimitStrength' not in runner.configured_options


def test_lc0_indices_and_history_match_reference():
    bindings = pytest.importorskip('lczero.backends')
    path = Path('ckpts/t1-256x10.pb.gz')
    if not path.exists():
        pytest.skip('Optional reference network not downloaded')
    backend = bindings.Backend(weights=bindings.Weights(str(path)), backend='blas')
    roots = [chess.STARTING_FEN, 'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1',
             '1r5k/P7/8/8/8/8/7p/K5R1 w - - 0 1', '1r5k/P7/8/8/8/8/7p/K5R1 b - - 0 1',
             '7k/8/8/3pP3/8/8/8/K7 w - d6 0 1']
    boards = [chess.Board(fen) for fen in roots]
    repeated = chess.Board()
    for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2:
        repeated.push_uci(move)
        boards.append(repeated.copy())
    rng, board = random.Random(9), chess.Board()
    for _ in range(500):
        if board.is_game_over():
            board.reset()
        boards.append(board.copy())
        board.push(rng.choice(list(board.legal_moves)))
    for board in boards:
        before = board.fen(), list(board.move_stack)
        state = bindings.GameState(fen=board.root().fen(), moves=[m.uci() for m in board.move_stack])
        assert {m.uci(): policy_index(m, board) for m in board.legal_moves} == dict(zip(state.moves(), state.policy_indices()))
        reference = state.as_input(backend)
        expected = np.array([[reference.val(p) if reference.mask(p) & (1 << sq) else 0
                              for sq in range(64)] for p in range(112)]).reshape(112, 8, 8)
        np.testing.assert_array_equal(encode_history(board), expected)
        assert (board.fen(), board.move_stack) == before


def test_mlx_export_prunes_unused_weights_and_quantizes_real_matmul(tmp_path):
    mx = pytest.importorskip('mlx.core')
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    from deploy.export_mlx import export_graph
    from play.mlx import Graph
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(64, 64)).astype(np.float16) * .05
    nodes = [helper.make_node('Reshape', ['input', 'shape'], ['flat']),
             helper.make_node('MatMul', ['flat', 'w'], ['test/policy']),
             helper.make_node('MatMul', ['flat', 'w'], ['test/value']),
             helper.make_node('Softmax', ['test/value'], ['test/wdl'], axis=1)]
    model = helper.make_model(helper.make_graph(nodes, 'test',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT16, ['batch', 112, 8, 8])],
        [helper.make_tensor_value_info('test/policy', TensorProto.FLOAT16, [None, 64]),
         helper.make_tensor_value_info('test/wdl', TensorProto.FLOAT16, [None, 64])],
        [numpy_helper.from_array(weights, 'w'), numpy_helper.from_array(np.array([-1, 64]), 'shape'),
         numpy_helper.from_array(np.ones(1000, np.float16), 'unused')]), opset_imports=[helper.make_opsetid('', 18)])
    source = tmp_path / 'graph.onnx'
    onnx.save(model, source)
    output = tmp_path / 'graph.mlx'
    stats = export_graph(source, output, bits=6)
    graph = Graph(output)
    assert stats['quantized_matrices'] == 1
    assert 'unused' not in graph.weights
    x = mx.ones((1, 112, 8, 8), dtype=mx.float16)
    result = graph.run(x, graph.weights)
    mx.eval(result)
    np.testing.assert_allclose(np.array(result[0]), np.ones((112, 64)) @ weights, atol=.035, rtol=.1)


def test_fused_attention_matches_explicit_attention():
    mx = pytest.importorskip('mlx.core')
    from play.mlx import fuse_attention
    nodes = [['MatMul', ['q', 'kt'], ['/mha/QK/product'], {}],
             ['Mul', ['/mha/QK/product', 'scale'], ['scaled'], {}],
             ['Add', ['scaled', 'mask'], ['masked'], {}],
             ['Softmax', ['masked'], ['prob'], {'axis': 3}],
             ['MatMul', ['prob', 'v'], ['out'], {}]]
    fused = fuse_attention(nodes, ['out'], {'scale': mx.array(.125)})
    assert len(fused) == 1 and fused[0][0] == 'Attention'
    mx.random.seed(3)
    q, k, v = [mx.random.normal((2, 4, 64, 64)) for _ in range(3)]
    mask = mx.random.normal((2, 4, 64, 64))
    explicit = mx.softmax(q @ k.swapaxes(-1, -2) * .125 + mask, axis=-1, precise=True) @ v
    fast = mx.fast.scaled_dot_product_attention(q, k, v, scale=fused[0][3]['scale'], mask=mask)
    np.testing.assert_allclose(np.array(fast), np.array(explicit), atol=1e-5, rtol=1e-4)


def test_compiled_graph_gradients_update_weights_without_teacher_gradients(tmp_path):
    mx = pytest.importorskip('mlx.core')
    import mlx.optimizers as optim
    from play.mlx import Graph
    from train.mlx import batch_rows, losses
    directory = tmp_path / 'train.mlx'
    directory.mkdir()
    value_weight = mx.zeros((64, 3)).at[:, 1].add(100.)
    weights = {'shape': mx.array([-1, 7168]), 'start': mx.array([0]), 'end': mx.array([64]),
               'axis': mx.array([1]), 'policy_w': mx.zeros((64, 1858)), 'value_w': value_weight}
    specification = {'inputs': ['input'], 'outputs': ['policy', 'wdl'],
                     'integer_constants': ['shape', 'start', 'end', 'axis'], 'nodes': [
        ['Reshape', ['input', 'shape'], ['flat'], {}],
        ['Slice', ['flat', 'start', 'end', 'axis'], ['features'], {}],
        ['MatMul', ['features', 'policy_w'], ['policy'], {}],
        ['MatMul', ['features', 'value_w'], ['value'], {}],
        ['Softmax', ['value'], ['wdl'], {'axis': 1}]]}
    (directory / 'graph.json').write_text(json.dumps(specification))
    mx.save_safetensors(str(directory / 'weights.safetensors'), weights)
    graph = Graph(directory)
    x, mask, target = batch_rows([{'fen': chess.STARTING_FEN, 'move': 'e2e4'}])
    gradient = mx.compile(mx.value_and_grad(lambda p: losses(graph, p, x, mask, target, target, mx.array([[1., 0., 0.]]))))
    before, grads = gradient(graph.weights)
    assert all(np.isfinite(np.array(g)).all() for g in grads.values())
    assert float(mx.abs(grads['value_w']).max()) > 0
    updated = optim.AdamW(.01).apply_gradients(grads, graph.weights)
    after, _ = gradient(updated)
    assert float(after[0]) < float(before[0])
    assert not np.array_equal(np.array(updated['policy_w']), np.array(graph.weights['policy_w']))


def test_activation_weighted_factorization_minimizes_its_actual_metric():
    from deploy.export_mlx import factorize
    rng = np.random.default_rng(9)
    weight = rng.normal(size=(64, 64))
    gram = np.diag(np.geomspace(.01, 100, 64))
    left, right, _, _ = factorize(weight, 16, gram)
    naive_left, naive_right, _, _ = factorize(weight, 16)
    metric = gram / np.mean(np.diag(gram)) + .01 * np.eye(64)
    error = weight - left @ right
    naive_error = weight - naive_left @ naive_right
    assert np.trace(error.T @ metric @ error) < np.trace(naive_error.T @ metric @ naive_error)
    assert np.linalg.matrix_rank(left @ right, tol=1e-8) == 16


def test_quantized_bottlenecks_support_both_matrix_orientations():
    mx = pytest.importorskip('mlx.core')
    from deploy.export_mlx import quantize_weights
    rng = np.random.default_rng(0)
    original = {'left': mx.array(rng.normal(size=(128, 16)).astype(np.float16) * .05),
                'right': mx.array(rng.normal(size=(16, 128)).astype(np.float16) * .05)}
    packed = dict(original)
    configurations = quantize_weights(packed, set(packed), 8)
    assert configurations['left'][2] and not configurations['right'][2]
    for name, weight in original.items():
        inputs = mx.array(rng.normal(size=(4, weight.shape[0])).astype(np.float16))
        group, bits, transpose = configurations[name]
        actual = mx.quantized_matmul(inputs, packed[name + '.packed'], packed[name + '.scales'],
                                    packed[name + '.biases'], group_size=group, bits=bits, transpose=transpose)
        np.testing.assert_allclose(np.array(actual), np.array(inputs @ weight), atol=.015, rtol=.1)
