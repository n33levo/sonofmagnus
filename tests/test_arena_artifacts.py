import sys

import pytest


@pytest.mark.parametrize('failure', ['changed_model', 'ply_cap'])
def test_arena_refuses_a_result_with_changed_artifacts_or_unfinished_games(tmp_path, monkeypatch, failure):
    import eval.arena as arena
    first, second = tmp_path / 'first.pt', tmp_path / 'second.pt'
    first.write_bytes(b'first model')
    second.write_bytes(b'second model')
    output = tmp_path / 'match.json'
    monkeypatch.setattr(sys, 'argv', ['arena', '--a', str(first), '--b', str(second), '--games', '2', '--out', str(output)])
    monkeypatch.setattr(arena, 'load_runner', lambda *args: object())
    def match(*args, **kwargs):
        if failure == 'changed_model':
            second.write_bytes(b'changed after loading')
        return {'terminations': {'ply_limit': 2} if failure == 'ply_cap' else {}}
    monkeypatch.setattr(arena, 'play_match', match)
    with pytest.raises(ValueError, match='changed' if failure == 'changed_model' else 'ply cap'):
        arena.main()
    assert not output.exists()
