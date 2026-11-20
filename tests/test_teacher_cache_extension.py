import hashlib
import json

import chess
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from model.artifacts import artifact_info
from scripts import extend_teacher_cache as extension
from train.dataset import ChessPositionDataset


def fixture(tmp_path, monkeypatch):
    teacher = tmp_path / 'teacher'
    teacher.write_text('fixed teacher')
    old, new, cache = [tmp_path / name for name in ('old.jsonl', 'new.jsonl', 'cache.npy')]
    rows = [{'fen': chess.STARTING_FEN, 'move': 'e2e4', 'weight': i + 1} for i in range(5)]
    old.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    new.write_text(''.join(json.dumps({**row, 'move': 'd2d4'}) + '\n' for row in rows + rows[:2]))
    # The fake teacher depends on batch size, exposing an incorrectly reused partial batch.
    def run(inputs, weights):
        return mx.full((len(inputs), 1858), len(inputs)), mx.full((len(inputs), 3), len(inputs))
    graph = type('Teacher', (), {'weights': {}, 'run': staticmethod(run)})()
    monkeypatch.setattr(extension, 'Graph', lambda _: graph)
    np.save(cache, np.concatenate((np.full((4, 1861), 4, np.float16), np.ones((1, 1861), np.float16))))
    dataset = ChessPositionDataset(old, augment=False)
    identity = {'teacher': artifact_info(teacher), 'data': artifact_info(old), 'samples': 5,
                'excluded_data': {}, 'runtime_source': {},
                'offsets_sha256': hashlib.sha256(dataset.offsets.tobytes()).hexdigest()}
    cache.with_suffix('.json').write_text(json.dumps({'identity': identity, 'artifact': artifact_info(cache)}))
    return teacher, old, new, cache


def test_extension_recomputes_partial_batch_and_accepts_changed_labels(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    output = tmp_path / 'extended.npy'
    manifest = extension.extend(*args, output, batch_size=4)
    assert manifest['reused_rows'] == 4 and manifest['verified_input_prefix_rows'] == 5
    np.testing.assert_array_equal(np.load(output), np.concatenate((np.full((4, 1861), 4), np.full((3, 1861), 3))))


def test_equal_fen_with_different_history_cannot_reuse_outputs(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in args[2].read_text().splitlines()]
    rows[0]['history'] = ['g1f3', 'g8f6', 'f3g1', 'f6g8']
    args[2].write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='Teacher input changed'):
        extension.extend(*args, tmp_path / 'bad.npy', batch_size=4)
    assert not (tmp_path / 'bad.npy').exists()
