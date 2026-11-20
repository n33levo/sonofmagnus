import pytest

from eval import sources


def runtime(tmp_path, monkeypatch):
    for package in ('play', 'chessio', 'model', 'eval'):
        (tmp_path / package).mkdir()
        (tmp_path / package / '__init__.py').write_text('value = 1\n')
    (tmp_path / 'eval/sources.py').write_text('source guard\n')
    monkeypatch.setattr(sources, 'ROOT', tmp_path)


def test_runtime_hashes_cover_each_inference_package(tmp_path, monkeypatch):
    runtime(tmp_path, monkeypatch)
    with sources.frozen_runtime('eval/__init__.py') as hashes:
        assert set(hashes) == {'play/__init__.py', 'chessio/__init__.py', 'model/__init__.py',
                              'eval/__init__.py', 'eval/sources.py'}


@pytest.mark.parametrize('change', ['modify', 'add', 'remove'])
def test_evaluation_rejects_runtime_edits_before_results_are_saved(tmp_path, monkeypatch, change):
    runtime(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='Runtime source changed'):
        with sources.frozen_runtime('eval/__init__.py'):
            path = tmp_path / 'play/__init__.py'
            if change == 'modify':
                path.write_text('value = 2\n')
            elif change == 'add':
                (tmp_path / 'play/new.py').write_text('value = 3\n')
            else:
                path.unlink()
