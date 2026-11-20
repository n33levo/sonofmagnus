import json
import numpy as np
import pytest

from eval.history_score import run
from model.artifacts import artifact_info


def history_inputs(tmp_path, loss=.1, cp=50):
    names = ('earlier', 'game1', 'game2')
    games = tmp_path / 'games.jsonl'
    games.write_text(''.join(json.dumps(dict(game_id=name, own_rating=2300, opponent_rating=2300,
        opponent=name, own_moves=1, score=0.)) + '\n' for name in names))
    calibration = tmp_path / 'calibration.json'
    calibration.write_text(json.dumps(dict(chronological_validation=dict(test_game_ids=['game1', 'game2'],
        training_game_ids=['earlier'],
        decision='retrospective-quality-signal-supported'), fitted_model=dict(weights=[0., -1., -1., -1., -1.],
        mean=[.1, .1, .1, .1], scale=[1., 1., 1., 1.]), inputs={str(games): artifact_info(games)})))
    paths = [tmp_path / 'base', tmp_path / 'candidate']
    for folder, point_loss, cp_loss in [(paths[0], .2, 150), (paths[1], loss, cp)]:
        folder.mkdir()
        (folder / 'review.json').write_text(json.dumps(dict(games=3, roots=3,
            protocol={'nodes': 100000}, summary={'candidate': {}})))
        for name in names:
            (folder / (name + '.json')).write_text(json.dumps([dict(game_id=name, ply=20, best=dict(cp=20, mate=None),
                scores=dict(candidate=dict(move='e2e4', point_loss=point_loss, cp_loss=cp_loss)))]))
    return games, calibration, *paths


@pytest.mark.parametrize('loss,cp,passes', [(.1, 50, True), (.3, 350, False)])
def test_paired_history_comparison_rewards_corrections_and_blocks_new_large_errors(tmp_path, loss, cp, passes):
    args = history_inputs(tmp_path, loss, cp)
    output = tmp_path / 'score.json'
    run(*args, output)
    report = json.loads(output.read_text())
    assert (report['all_games'], report['all_roots'], report['games'], report['roots']) == (3, 3, 2, 2)
    earlier = report['game_diagnostics'][0]
    assert earlier['split'] == 'earlier-training' and earlier['actual_score'] == 0.
    assert earlier['candidate']['worst_move']['move'] == 'e2e4'
    assert all(report['checks'].values()) == passes
    assert not report['promoted'] and not report['automatic_rated']
    if passes:
        assert report['candidate_quality_index'] > report['baseline_quality_index']
        np.testing.assert_allclose(report['mean_point_loss_reduction_per_game'], .1)
    else:
        assert len(report['new_300cp_errors']) == 2


@pytest.mark.parametrize('change', ['missing-root', 'different-reference', 'stale-games', 'split-overlap'])
def test_scoring_checks_earlier_games_and_calibration_snapshot(tmp_path, change):
    args = history_inputs(tmp_path)
    if change == 'stale-games':
        args[0].write_text(args[0].read_text().replace('2300', '2200'))
    elif change == 'split-overlap':
        calibration = json.loads(args[1].read_text())
        calibration['chronological_validation']['training_game_ids'].append('game1')
        args[1].write_text(json.dumps(calibration))
    else:
        path = args[3] / 'earlier.json'
        rows = json.loads(path.read_text())
        if change == 'missing-root':rows = []
        else:rows[0]['best']['cp'] = 999
        path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        run(*args, tmp_path / 'score.json')
