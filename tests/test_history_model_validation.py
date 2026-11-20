import copy
import json

import pytest

from eval.calibrate_lichess_history import FEATURES
from eval.history_model_validation import compare, run
from model.artifacts import artifact_info


def fixture():
    games, quality, assignments = [], [], []
    for i in range(20):
        key, score = str(i), (i % 3) / 2
        games.append(dict(game_id=key, score=score, opponent='opponent' + str(i % 5),
            own_rating=2300, opponent_rating=2300, rated=True, perf='blitz', rating_diff=1,
            created_ms=i * 100, finished_ms=i * 100 + 50))
        quality.append(dict(game_id=key, actual_score=score,
            split='earlier-fitting' if i < 15 else 'later-validation',
            features=dict(zip(FEATURES, [.1 * (1 - score), .2 * (1 - score), 0., 0.]))))
        assignments.append(dict(game_id=key, score=score, opponent=games[-1]['opponent'],
            opponent_rating=2300, model_sha256=str(i % 3) * 64, attribution='saved-review'))
    return games, quality, assignments


def test_tested_checkpoint_and_later_outcomes_never_enter_fitting():
    data = fixture()
    report = compare(*data)
    for fold in report['folds']:
        assert set(fold['training_game_ids']).isdisjoint(fold['tested_game_ids'])
        assert all(int(key) < 15 and str(int(key) % 3) * 64 != fold['model'] for key in fold['training_game_ids'])
    changed = copy.deepcopy(data)
    for i in range(15, 20):
        for row, field in [(changed[0][i], 'score'), (changed[1][i], 'actual_score'), (changed[2][i], 'score')]:
            row[field] = 1 - row[field]
    revised = compare(*changed)
    assert [row['quality_estimate'] for row in report['game_predictions']] == [row['quality_estimate'] for row in revised['game_predictions']]
    assert report['cohorts']['later-validation']['quality_squared_error'] < report['cohorts']['later-validation']['rating_only_squared_error']
    assert report['target_2800_offline_threshold'] is None


def test_unknown_models_and_non_rating_games_stay_visible_but_are_not_fitted():
    data = fixture()
    data[2][0].update(model_sha256=None, attribution='unknown')
    data[0][1]['rating_diff'] = None
    data[1][1]['split'] = 'missing-rating-delta'
    report = compare(*data)
    assert report['all_games'] == 20 and report['scored_games'] == 18
    assert {row['game_id'] for row in report['excluded_games']} == {'0', '1'}


def test_cli_binds_saved_reviews_to_the_snapshot_and_rejects_changed_evidence(tmp_path):
    games, quality, assignments = fixture()
    source = tmp_path / 'games.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in games))
    inputs = {str(source): artifact_info(source)}
    validation, attribution, output = [tmp_path / name for name in ('validation.json', 'attribution.json', 'output.json')]
    validation.write_text(json.dumps(dict(game_diagnostics=quality, inputs=inputs)))
    attribution.write_text(json.dumps(dict(game_details=assignments, inputs=inputs)))
    run(source, validation, attribution, output)
    report = json.loads(output.read_text())
    assert report['scored_games'] == 20
    assert report['inputs'][str(source)] == inputs[str(source)]
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError, match='input changed'):
        run(source, validation, attribution, tmp_path / 'invalid.json')
    assert not (tmp_path / 'invalid.json').exists()


@pytest.mark.parametrize('change', ['duplicate', 'outcome', 'missing', 'cohort', 'time', 'nonfinite'])
def test_mismatched_or_leaking_observations_are_rejected(change):
    data = fixture()
    if change == 'duplicate': data[2].append(data[2][0])
    elif change == 'outcome': data[2][0]['score'] = .5
    elif change == 'missing': data[1].pop()
    elif change == 'cohort': data[1][0]['split'] = 'missing-rating-delta'
    elif change == 'time': data[0][0]['finished_ms'] = 1600
    else: data[1][0]['features'][FEATURES[0]] = float('nan')
    with pytest.raises(ValueError):
        compare(*data)
