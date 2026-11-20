import json

import chess
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from chessio.policy_map import move_to_policy_index
from play.chessbench import ChessBenchRunner, action_vocabulary, tokenize


def fixture(tmp_path):
    rng = np.random.default_rng(31)
    shapes = {'embed/embeddings': (1968, 8), 'embed_1/embeddings': (79, 8),
              'linear/w': (8, 32), 'linear_1/w': (8, 32), 'linear_2/w': (32, 8),
              'linear_3/w': (8, 128), 'linear_3/b': (128,)}
    for i in range(4):
        name = 'linear' + (f'_{i}' if i else '')
        shapes['multi_head_dot_product_attention/' + name + '/w'] = (8, 8)
    weights = {name: mx.array(rng.normal(0, .03, shape).astype(np.float16)) for name, shape in shapes.items()}
    for i in range(3):
        name = 'layer_norm' + (f'_{i}' if i else '')
        weights[name + '/scale'], weights[name + '/offset'] = mx.ones((8,), dtype=mx.float16), mx.zeros((8,), dtype=mx.float16)
    (tmp_path / 'metadata.json').write_text(json.dumps({'architecture': 'chessbench-action-value',
        'dimension': 8, 'layers': 1, 'heads': 2, 'quantized': {}}))
    mx.save_safetensors(str(tmp_path / 'weights.safetensors'), weights)
    return ChessBenchRunner(tmp_path)


def test_tokens_and_action_indices_match_the_published_order():
    actions = action_vocabulary()
    assert len(actions) == 1968 and len(set(actions.values())) == 1968
    assert actions['a1b1'] == 0 and actions['a2a1q'] == 1792 and actions['h7g8n'] == 1967
    tokens = tokenize(chess.STARTING_FEN)
    assert tokens.shape == (77,) and tokens.dtype == np.int32 and tokens[0] == 29
    assert tokens[-6:].tolist() == [0, 30, 30, 1, 30, 30]
    with pytest.raises(ValueError, match='three-digit'):
        tokenize(chess.STARTING_FEN.replace('0 1', '0 1000'))


def test_one_root_forward_queries_legal_actions_without_successor_boards(tmp_path, monkeypatch):
    runner = fixture(tmp_path)
    board = chess.Board()
    before = board.fen(), list(board.move_stack)
    received, forward = [], runner.run
    def record(tokens, weights):
        received.append(np.array(tokens))
        return forward(tokens, weights)
    runner.run = record
    monkeypatch.setattr(board, 'push', lambda move: pytest.fail('Successor board evaluated'))
    move = runner.select_move(board, deterministic=True)
    assert move in board.legal_moves and (board.fen(), board.move_stack) == before
    assert runner.forward_calls == runner.root_positions == 1 and runner.action_queries == 20
    tokens = received[0]
    assert tokens.shape == (20, 79) and (tokens[:, 0] == 0).all()
    np.testing.assert_array_equal(tokens[:, 1:78], np.tile(tokenize(board.fen()), (20, 1)))
    assert tokens[:, -1].tolist() == [runner.actions[m.uci()] for m in sorted(board.legal_moves, key=lambda m: m.uci())]


def test_bidirectional_transformer_matches_independent_numpy_computation(tmp_path):
    runner = fixture(tmp_path)
    runner.weights = {name: value.astype(mx.float32) for name, value in runner.weights.items()}
    p = {name: np.array(value) for name, value in runner.weights.items()}
    tokens = np.stack([np.r_[0, tokenize(chess.STARTING_FEN), a] for a in (runner.actions['e2e4'], runner.actions['g1f3'])])
    def norm(x, index):
        name = 'layer_norm' + (f'_{index}' if index else '')
        return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * p[name + '/scale'] + p[name + '/offset']
    h = p['embed/embeddings'][tokens] * np.sqrt(np.float32(8)) + p['embed_1/embeddings']
    a = norm(h, 0)
    q, k, v = [(a @ p['multi_head_dot_product_attention/' + ('linear' + (f'_{i}' if i else '')) + '/w']).reshape(2, 79, 2, 4).transpose(0, 2, 1, 3) for i in range(3)]
    scores = q @ k.swapaxes(-1, -2) / 2
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    h += (probs @ v).transpose(0, 2, 1, 3).reshape(2, 79, 8) @ p['multi_head_dot_product_attention/linear_3/w']
    a = norm(h, 1)
    gate = a @ p['linear/w']
    h += (gate / (1 + np.exp(-gate)) * (a @ p['linear_1/w'])) @ p['linear_2/w']
    expected = norm(h[:, -1], 2) @ p['linear_3/w'] + p['linear_3/b']
    actual = runner.run(mx.array(tokens), runner.weights)
    np.testing.assert_allclose(np.array(actual), expected, atol=2e-6, rtol=1e-4)


def test_equal_values_use_upstream_lexical_ties_and_terminals_need_no_forward(tmp_path):
    runner = fixture(tmp_path)
    runner.run = lambda tokens, weights: mx.zeros((len(tokens), 128))
    assert runner.select_move(chess.Board(), deterministic=True).uci() == 'a2a3'
    probabilities = runner.move_probs(chess.Board())
    assert max(probabilities, key=probabilities.get).uci() == 'a2a3'
    assert runner.forward_calls == 2
    terminal = chess.Board('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1')
    assert runner.select_move(terminal, deterministic=True) is None and runner.forward_calls == 2


def test_large_residuals_remain_finite_and_match_full_precision(tmp_path):
    runner = fixture(tmp_path)
    runner.weights['layer_norm_1/offset'] = mx.full((8,), 10, dtype=mx.float16)
    runner.weights['linear/w'] = mx.ones((8, 32), dtype=mx.float16)
    runner.weights['linear_1/w'] = runner.weights['linear/w']
    runner.weights['linear_2/w'] = mx.broadcast_to(
        mx.array([20000, -20000] * 4, dtype=mx.float16)[None, :], (32, 8))
    tokens = mx.array(np.r_[0, tokenize(chess.STARTING_FEN), runner.actions['e2e4']][None, :])
    actual = np.array(runner.run(tokens, runner.weights))
    full = {name: value.astype(mx.float32) for name, value in runner.weights.items()}
    expected = np.array(runner.run(tokens, full))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-2)
