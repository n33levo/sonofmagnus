import copy
import json
from pathlib import Path

import numpy as np
import pytest

from eval.calibrate_lichess_history import FEATURES, fit, game_features
from eval.history_extension import compare, run
from model.artifacts import artifact_info


def fixture():
    games, roots = [], {}
    for i in range(22):
        score = (i % 3) / 2
        game = dict(game_id=str(i), url='https://lichess.org/' + str(i), score=score,
            opponent='bot' + str(i % 5), own_rating=2300, opponent_rating=2200 + i * 5,
            rated=True, perf='blitz', rating_diff=1, own_moves=2,
            created_ms=i * 100, finished_ms=i * 100 + 50)
        games.append(game)
        roots[str(i)] = [dict(game_id=str(i), ply=ply, best=dict(cp=0, mate=None),
            scores=dict(played=dict(move=move, point_loss=(1 - score) * loss, cp_loss=cp)))
            for ply, move, loss, cp in [(0, 'e2e4', .1, 50), (2, 'g1f3', .2, 200)]]
    original = copy.deepcopy(games[:18])
    features = {key: game_features(rows) for key, rows in roots.items()}
    training, tested = [str(i) for i in range(12)], [str(i) for i in range(12, 18)]
    x = np.stack([features[key] for key in training])
    y = np.array([games[int(key)]['score'] for key in training])
    offset = np.array([games[int(key)]['own_rating'] - games[int(key)]['opponent_rating']
                       for key in training]) * np.log(10) / 400
    calibration = dict(fitted_model=fit(x, y, offset),
        chronological_validation=dict(training_game_ids=training, test_game_ids=tested),
        game_features=[dict(game_id=row['game_id'], score=row['score'], opponent=row['opponent'],
            opponent_rating=row['opponent_rating'], features=dict(zip(FEATURES, features[row['game_id']])))
            for row in original])
    return games, original, roots, calibration


def test_new_outcomes_do_not_change_predictions_or_enter_fitting():
    data = fixture()
    result = compare(*data)
    for row in data[0][18:]:
        row['score'] = 1 - row['score']
    changed = compare(*data)
    assert [r['quality_estimate'] for r in result['game_diagnostics']] == [
        r['quality_estimate'] for r in changed['game_diagnostics']]
    assert result['games'] == 22 and result['decisions'] == 44
    assert result['new_rated_games'] == 4
    assert result['calibration_refitted_on_new_games'] is False
    assert result['offline_elo_conversion_established'] is False


def test_non_rating_games_still_have_quality_and_outcomes():
    data = fixture()
    data[0][-1]['rating_diff'] = None
    result = compare(*data)
    assert result['games'] == 22 and result['new_rated_games'] == 3
    assert result['new_games_excluded_from_rating_score'] == ['21']
    assert len(result['game_diagnostics']) == 22


@pytest.mark.parametrize('change', ['duplicate', 'changed_old_game', 'missing_quality',
    'duplicate_root', 'changed_feature', 'leaking_time', 'leaking_split', 'nonfinite', 'weights'])
def test_invalid_or_leaking_evidence_is_rejected(change):
    data = fixture()
    games, _, roots, calibration = data
    if change == 'duplicate': games.append(games[-1])
    elif change == 'changed_old_game': games[0]['score'] = 1
    elif change == 'missing_quality': roots.pop('21')
    elif change == 'duplicate_root': roots['21'][1]['ply'] = 0
    elif change == 'changed_feature': roots['0'][0]['scores']['played']['point_loss'] = .9
    elif change == 'leaking_time': games[18]['created_ms'] = 1700
    elif change == 'leaking_split': calibration['chronological_validation']['training_game_ids'].append('18')
    elif change == 'nonfinite': roots['21'][0]['scores']['played']['point_loss'] = float('nan')
    else: calibration['fitted_model']['weights'][0] += .1
    with pytest.raises(ValueError):
        compare(*data)


def test_file_review_binds_all_histories_actions_and_teacher_losses(tmp_path):
    games, original, roots, calibration = fixture()
    snapshot, registered = tmp_path / 'snapshot', tmp_path / 'registered'
    snapshot.mkdir()
    registered.mkdir()
    positions = [dict(game_id=key, ply=row['ply'], played_move=row['scores']['played']['move'],
                      fen='full-history-' + key, history=['a2a4'] * row['ply'])
                 for key, rows in roots.items() for row in rows]
    for folder, selected, members in ((snapshot, games, positions), (registered, original, positions[:36])):
        for name, values in [('games.jsonl', selected), ('positions.jsonl', members)]:
            (folder / name).write_text(''.join(json.dumps(row) + '\n' for row in values))
    quality = [tmp_path / 'old-quality', tmp_path / 'new-quality']
    for folder, selected, members in ((quality[0], original, positions[:36]), (quality[1], games[18:], positions[36:])):
        folder.mkdir()
        source = folder / 'positions.jsonl'
        source.write_text(''.join(json.dumps(row) + '\n' for row in members))
        for game in selected:
            rows = copy.deepcopy(roots[game['game_id']])
            for row in rows:
                row['best']['points'] = .5
                played = row['scores']['played']
                played['observed'] = dict(cp=-played['cp_loss'], mate=None, points=.5 - played['point_loss'])
                played['point_loss'] = .5 - played['observed']['points']
                played['forced_exceeds_best'] = False
                roots[game['game_id']] = rows
            (folder / (game['game_id'] + '.json')).write_text(json.dumps(rows))
        (folder / 'review.json').write_text(json.dumps(dict(positions=str(source), roots=len(members),
            protocol=dict(positions=artifact_info(source), nodes=100000,
                producer=artifact_info(Path(__file__).parents[1] / 'eval/lichess_quality.py')))))
    # Rebuild the frozen calibration using the exact serialized arithmetic.
    for row in calibration['game_features']:
        row['features'] = dict(zip(FEATURES, game_features(roots[row['game_id']]).tolist()))
    keys = calibration['chronological_validation']['training_game_ids']
    calibration['fitted_model'] = {key: value.tolist() for key, value in fit(
        np.stack([game_features(roots[key]) for key in keys]),
        np.array([games[int(key)]['score'] for key in keys]),
        np.array([games[int(key)]['own_rating'] - games[int(key)]['opponent_rating'] for key in keys])
        * np.log(10) / 400).items()}
    calibration['inputs'] = {str(registered / 'games.jsonl'): artifact_info(registered / 'games.jsonl')}
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps(calibration))
    run(snapshot, registered, path, quality, tmp_path / 'score.json')
    assert json.loads((tmp_path / 'score.json').read_text())['decisions'] == 44
    bad = quality[1] / '21.json'
    saved = bad.read_text()
    revised = json.loads(saved)
    revised[0]['scores']['played']['point_loss'] += .1
    bad.write_text(json.dumps(revised))
    with pytest.raises(ValueError, match='teacher observations'):
        run(snapshot, registered, path, quality, tmp_path / 'invalid.json')
    bad.write_text(saved)
    revised = json.loads(saved)
    revised[0]['scores']['played']['move'] = 'a2a3'
    bad.write_text(json.dumps(revised))
    with pytest.raises(ValueError, match='real played move'):
        run(snapshot, registered, path, quality, tmp_path / 'invalid.json')
    assert not (tmp_path / 'invalid.json').exists()
