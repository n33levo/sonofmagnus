import json

import pytest

from model.artifacts import artifact_info
from scripts.evaluation_recovery import copy_evaluation_requests


def fixture(tmp_path, flags=None, exclusions=None):
    path = tmp_path / 'original.jsonl'
    row = dict(validation_only=True, diagnostic_only=True, policy_supervision=False,
               fen='unchanged', history=['e2e4'], actor_choices=['e7e5'])
    row.update(flags or {})
    path.write_text(json.dumps(row) + '\n')
    meta = dict(artifact=artifact_info(path), inputs={}, models={'source': 'retained'})
    if exclusions is not None:
        meta['exclusions'] = exclusions
    path.with_suffix('.json').write_text(json.dumps(meta))
    return path


def test_adds_only_evaluation_metadata_and_preserves_query_bytes(tmp_path):
    original = fixture(tmp_path)
    metadata = original.with_suffix('.json').read_bytes()
    output = tmp_path / 'repaired.jsonl'
    result = copy_evaluation_requests(original, output)
    assert original.read_bytes() == output.read_bytes()
    assert original.with_suffix('.json').read_bytes() == metadata
    repaired = json.loads(output.with_suffix('.json').read_text())
    assert repaired['models'] == {'source': 'retained'} and repaired['exclusions'] == []
    assert not result['approved_for_fit']


@pytest.mark.parametrize('flags', [{'policy_supervision': True}, {'diagnostic_only': False},
                                  {'validation_only': False}])
def test_training_or_ambiguous_queries_cannot_drop_exclusions(tmp_path, flags):
    original = fixture(tmp_path, flags)
    with pytest.raises(ValueError):
        copy_evaluation_requests(original, tmp_path / 'repaired.jsonl')


@pytest.mark.parametrize('exclusions', [[], ['protected.jsonl']])
def test_existing_exclusions_cannot_be_overridden(tmp_path, exclusions):
    original = fixture(tmp_path, exclusions=exclusions)
    with pytest.raises(ValueError):
        copy_evaluation_requests(original, tmp_path / 'repaired.jsonl')


def test_changed_input_cannot_be_relabelled(tmp_path):
    original = fixture(tmp_path)
    original.write_bytes(original.read_bytes() + b'\n')
    with pytest.raises(ValueError):
        copy_evaluation_requests(original, tmp_path / 'repaired.jsonl')
