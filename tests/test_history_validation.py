import json
from pathlib import Path

import pytest

from eval.history_validation import run, validate_split
from model.artifacts import artifact_info
from eval.calibrate_lichess_history import run as calibrate


def records():
    return [dict(game_id=str(i), rated=True, perf='blitz', rating_diff=1,
                 created_ms=i * 100, finished_ms=i * 100 + 50) for i in range(4)]


def test_missing_delta_games_are_retained_outside_the_calibration_split():
    games = records() + [dict(game_id='extra', rated=True, perf='blitz', rating_diff=None)]
    train, test = validate_split(games, dict(training_game_ids=['0', '1'], test_game_ids=['2', '3']))
    assert [row['game_id'] for row in train + test] == ['0', '1', '2', '3']


@pytest.mark.parametrize('change', ['overlap', 'missing-game', 'duplicate-game', 'overlapping-time'])
def test_validation_rejects_leakage_or_incomplete_calibration_cohort(change):
    games = records()
    split = dict(training_game_ids=['0', '1'], test_game_ids=['2', '3'])
    if change == 'overlap':split['test_game_ids'][0] = '1'
    elif change == 'missing-game':split['test_game_ids'].pop()
    elif change == 'duplicate-game':games.append(games[0])
    else:games[1]['finished_ms'] = 201
    with pytest.raises(ValueError):
        validate_split(games, split)


def test_validation_rejects_a_substituted_snapshot(tmp_path):
    folder, quality = tmp_path / 'games', tmp_path / 'quality'
    folder.mkdir()
    quality.mkdir()
    for path in (folder / 'games.jsonl', folder / 'positions.jsonl', quality / 'review.json'):
        path.write_text('{}')
    trusted = tmp_path / 'trusted'
    trusted.mkdir()
    (trusted / 'games.jsonl').write_text('{"different":true}')
    calibration = tmp_path / 'calibration.json'
    calibration.write_text(json.dumps(dict(inputs={str(path): artifact_info(path) for path in
        (trusted / 'games.jsonl', folder / 'positions.jsonl', quality / 'review.json')})))
    with pytest.raises(ValueError, match='another history snapshot'):
        run(folder, quality, calibration, tmp_path / 'report.json')


@pytest.mark.parametrize('legacy', [False, True])
def test_frozen_calibration_validates_from_another_checkout(tmp_path, monkeypatch, legacy):
    project = tmp_path / 'project'
    folder, quality = project / 'data/games', project / 'out/quality'
    folder.mkdir(parents=True)
    quality.mkdir(parents=True)
    games = []
    for i, score in enumerate([1., .5, 0., 1., 0., .5]):
        game = dict(game_id=str(i), rated=True, perf='blitz', rating_diff=1, own_moves=1,
            created_ms=i * 100, finished_ms=i * 100 + 50, score=score, own_rating=2300,
            opponent_rating=2300, opponent='opponent' + str(i), url='https://lichess.org/' + str(i))
        games.append(game)
        root = dict(game_id=str(i), ply=0,
            scores=dict(played=dict(move='e2e4', point_loss=.1 * (1 - score), cp_loss=100 * (1 - score))))
        (quality / (str(i) + '.json')).write_text(json.dumps([root]))
    (folder / 'games.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in games))
    (folder / 'positions.jsonl').write_text('{}\n')
    (quality / 'review.json').write_text(json.dumps(dict(games=6, roots=6, summary={'played': {}},
        protocol=dict(positions=artifact_info(folder / 'positions.jsonl')))))
    monkeypatch.chdir(project)
    calibration = project / 'out/calibration.json'
    calibrate(Path('data/games'), Path('out/quality'), calibration)
    saved = json.loads(calibration.read_text())
    assert saved['input_root'] == str(project)
    if legacy:
        saved.pop('input_root')
        calibration.write_text(json.dumps(saved))
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'validation.json'
    run(folder, quality, calibration, output)
    assert json.loads(output.read_text())['later_games'] == 2
