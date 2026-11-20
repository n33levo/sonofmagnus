import json
import sys

import chess
import pytest


def test_new_training_resets_source_progress_before_reference_caching(tmp_path, monkeypatch):
    mx = pytest.importorskip('mlx.core')
    from train import mlx
    source, output = tmp_path/'source', tmp_path/'new-run'
    source.mkdir()
    (source/'graph.json').write_text('{}')
    original = dict(training_samples_processed=999, training_epochs_completed=7)
    (source/'metadata.json').write_text(json.dumps(original))
    data = tmp_path/'data.jsonl'
    data.write_text(json.dumps(dict(fen=chess.STARTING_FEN, move='e2e4'))+'\n')

    class Graph:
        def __init__(self, *args):
            self.weights = {'/policy/w':mx.ones((2, 3))}
            self.nodes = [('MatMul', ['x', '/policy/w'], ['policy'], {})]
            self.quantized = False

    class CacheReached(Exception):
        pass

    def cache(*args):
        current = json.loads((output/'metadata.json').read_text())
        assert current['training_samples_processed'] == 0
        assert current['training_epochs_completed'] == 0
        assert json.loads((source/'metadata.json').read_text()) == original
        raise CacheReached

    monkeypatch.setattr(mlx, 'Graph', Graph)
    monkeypatch.setattr(mlx, 'cache_teacher', cache)
    monkeypatch.setattr(sys, 'argv', ['train.mlx', '--student', str(source), '--teacher', str(source),
        '--teacher-cache', str(tmp_path/'reference.npy'), '--data', str(data), '--out', str(output), '--epochs', '1'])
    with pytest.raises(CacheReached):
        mlx.main()
