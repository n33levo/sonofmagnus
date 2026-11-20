import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from train.mlx import losses, teacher_fractions


def test_deep_live_feedback_uses_tactical_anchor_without_changing_other_rows():
    rows = [{'source': 'lichess-live-stockfish-hard-feedback'},
            {'source': 'lichess-puzzle-stockfish-40-meganodes'}, {'source': 'lichess-master-history-stockfish18'}]
    np.testing.assert_allclose(np.array(teacher_fractions(rows, .8, .05)), [.05, .05, .8])


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


def test_partial_policy_learns_recorded_ranking_without_penalizing_unknown_moves():
    class Graph:
        def forward(self, inputs, weights, raw_value=False):
            return weights['policy'], mx.zeros((3, 3))
    weights = {'policy': mx.zeros((3, 4))}
    mask = mx.ones((3, 4), dtype=mx.bool_)
    target = mx.array([[.75, .25, 0., 0.], [1., 0., 0., 0.], [.5, .25, .125, .125]])
    def objective(parameters, conditional):
        return losses(Graph(), parameters, mx.zeros((3, 1)), mask, target, target,
                      mx.zeros((3, 3)), value_weight=0, teacher_weight=0, conditional_policy=conditional)[0]
    conditional, grad = mx.value_and_grad(objective)(weights, True)
    full, original_grad = mx.value_and_grad(objective)(weights, False)
    actual = np.array(grad['policy'])
    np.testing.assert_allclose(actual[0], [-1/12, 1/12, 0, 0], atol=1e-7)
    np.testing.assert_array_equal(actual[1:], np.array(original_grad['policy'])[1:])
    assert float(conditional) == pytest.approx((np.log(2) + 2*np.log(4))/3)
    assert float(full) == pytest.approx(np.log(4))


@pytest.mark.parametrize('feature_cache', [False, True])
@pytest.mark.parametrize('tune_last_blocks', [0, 1])
def test_policy_cli_anchors_to_teacher_and_freezes_untuned_weights(tmp_path, monkeypatch, feature_cache, tune_last_blocks):
    import chess
    import json
    from model.artifacts import artifact_info
    from train.mlx import main

    source, output = tmp_path / 'source', tmp_path / 'trained'
    source.mkdir()
    rng = np.random.default_rng(14)
    weights = {'shape': mx.array([-1, 7168]),
               '/encoder0/w': mx.array(rng.normal(0, .01, (7168, 4)).astype(np.float16)),
               '/policy/w': mx.array(rng.normal(0, .01, (4, 1858)).astype(np.float16)),
               '/value/w': mx.array(rng.normal(0, .01, (4, 3)).astype(np.float16))}
    nodes = [['Reshape', ['board', 'shape'], ['flat'], {}],
             ['MatMul', ['flat', '/encoder0/w'], ['features'], {}],
             ['MatMul', ['features', '/policy/w'], ['policy'], {}],
             ['MatMul', ['features', '/value/w'], ['value_logits'], {}],
             ['Softmax', ['value_logits'], ['wdl'], {'axis': -1}]]
    (source / 'graph.json').write_text(json.dumps({'inputs': ['board'], 'outputs': ['policy', 'wdl'],
                                                 'integer_constants': ['shape'], 'nodes': nodes}))
    (source / 'metadata.json').write_text('{}')
    mx.save_safetensors(str(source / 'weights.safetensors'), weights)
    frozen = artifact_info(source)
    board = chess.Board()
    rows = []
    for move in ('e2e4', 'e7e5', 'g1f3'):
        rows.append({'fen': board.fen(), 'move': move})
        board.push_uci(move)
    data = tmp_path / 'rows.jsonl'
    data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    excluded = tmp_path / 'puzzle.jsonl'
    excluded.write_text(json.dumps({'fen': chess.STARTING_FEN, 'moves': ['e2e4']}) + '\n')
    arguments = ['train', '--student', str(source), '--teacher', str(source),
        '--data', str(data), '--out', str(output), '--mode', 'policy', '--epochs', '1',
        '--batch-size', '2', '--amp', 'fp16' if feature_cache else 'none', '--teacher-weight', '.8', '--teacher-temperature', '1',
        '--exclude-data', str(excluded), '--tune-last-blocks', str(tune_last_blocks)]
    if feature_cache:
        arguments += ['--feature-cache', str(tmp_path / 'features')]
    monkeypatch.setattr('sys.argv', arguments)
    main()
    actual = mx.load(str(output / 'weights.safetensors'))
    assert artifact_info(source) == frozen
    for name in ('/value/w', *(() if tune_last_blocks else ('/encoder0/w',))):
        np.testing.assert_array_equal(np.array(actual[name]), np.array(weights[name]))
    if tune_last_blocks:
        assert not np.array_equal(np.array(actual['/encoder0/w']), np.array(weights['/encoder0/w']))
    assert not np.array_equal(np.array(actual['/policy/w']), np.array(weights['/policy/w']))
    metadata = json.loads((output / 'metadata.json').read_text())
    assert metadata['teacher_source'] == frozen and metadata['trainable_scalars'] == 4 * 1858 + tune_last_blocks * 7168 * 4
    expected_prefixes = ['/policy/', '/encoder0/'] if tune_last_blocks else ['/policy/']
    assert metadata['trainable_prefixes'] == expected_prefixes
    assert '0.8 frozen teacher CE' in metadata['objective']
    metrics = json.loads((output / 'train.jsonl').read_text())
    assert metadata['heldout_excluded'] == 2 and metadata['excluded_data'][str(excluded)] == artifact_info(excluded)
    assert metrics['samples'] == 1 and metrics['value_loss'] == 0
    if feature_cache:
        assert metadata['feature_cache'] == artifact_info(tmp_path / 'features.json')
        cached = json.loads((tmp_path / 'features.json').read_text())
        assert cached['identity']['excluded_data'] == metadata['excluded_data']
        assert cached['identity']['runtime_source']['train/policy_cache.py']
