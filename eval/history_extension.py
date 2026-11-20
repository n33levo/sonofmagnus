"""Score new public games with the frozen historical outcome model."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from eval.calibrate_lichess_history import FEATURES, cluster_interval, fit, game_features, predict, sigmoid
from eval.lichess_quality import loss
from model.artifacts import artifact_info


def compare(games, original, roots, calibration):
    records = {row['game_id']: row for row in games}
    previous = {row['game_id']: row for row in original}
    if (len(records) != len(games) or len(previous) != len(original)
            or not previous.keys() < records.keys() or set(roots) != set(records)
            or any(records[key] != row for key, row in previous.items())):
        raise ValueError('Require an append-only snapshot and quality for every game')
    split = calibration['chronological_validation']
    training, tested = split['training_game_ids'], split['test_game_ids']
    cohort = {key for key, row in previous.items() if row['rated'] and row['perf'] == 'blitz'
              and row['rating_diff'] is not None}
    if (not training or not tested or len(set(training + tested)) != len(training + tested)
            or set(training + tested) != cohort):
        raise ValueError('Frozen calibration split does not cover the original cohort')
    added = sorted(records.keys() - previous.keys(), key=lambda key: (records[key]['created_ms'], key))
    if (max(previous[key]['finished_ms'] for key in training) >= min(previous[key]['created_ms'] for key in tested)
            or max(row['finished_ms'] for row in original) >= min(records[key]['created_ms'] for key in added)):
        raise ValueError('New scoring games must follow the entire original snapshot')
    features = {}
    for key, game in records.items():
        rows = roots[key]
        if (not rows or len(rows) != game['own_moves'] or any(row['game_id'] != key for row in rows)
                or [row['ply'] for row in rows] != sorted({row['ply'] for row in rows})):
            raise ValueError('Incomplete or duplicated quality observations')
        values = game_features(rows)
        if (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
                or game['score'] not in (0., .5, 1.)):
            raise ValueError('Invalid game features or outcome')
        features[key] = values
    saved = {row['game_id']: row for row in calibration['game_features']}
    if len(saved) != len(training + tested) or set(saved) != cohort:
        raise ValueError('Incomplete frozen calibration features')
    for key in cohort:
        row, stored = records[key], saved[key]
        if (any(row[name] != stored[name] for name in ('score', 'opponent', 'opponent_rating'))
                or not np.array_equal(features[key], [stored['features'][name] for name in FEATURES])):
            raise ValueError('Original quality or outcome changed')

    def arrays(keys):
        return (np.stack([features[key] for key in keys]),
                np.array([records[key]['score'] for key in keys]),
                np.array([records[key]['own_rating'] - records[key]['opponent_rating'] for key in keys])
                * np.log(10) / 400)

    x, y, offset = arrays(training)
    model = {key: np.array(value) for key, value in calibration['fitted_model'].items()}
    rebuilt = fit(x, y, offset)
    if any(not np.allclose(model[key], value, rtol=0, atol=1e-12) for key, value in rebuilt.items()):
        raise ValueError('Frozen model does not reproduce from earlier games')
    baseline = fit(np.zeros((len(training), 0)), y, offset)
    eligible = [key for key in added if records[key]['rated'] and records[key]['perf'] == 'blitz'
                and records[key]['rating_diff'] is not None]
    if not eligible:
        raise ValueError('No new rated-blitz outcome to score')
    x, y, offset = arrays(eligible)
    estimated = predict(model, x, offset)
    reference = predict(baseline, np.zeros((len(eligible), 0)), offset)
    gain = (reference - y) ** 2 - (estimated - y) ** 2
    details = []
    predictions = {key: (float(value), float(base)) for key, value, base in zip(eligible, estimated, reference)}
    for key, game in records.items():
        rows = roots[key]
        recoverable = [row for row in rows if ((row['best'].get('mate') or 0) > 0
                       or (row['best']['cp'] is not None and row['best']['cp'] >= -300))]
        errors = [row for row in recoverable if row['scores']['played']['point_loss'] >= .3]
        first = errors[0] if errors else None
        details.append(dict(game_id=key, url=game['url'], score=game['score'], opponent=game['opponent'],
            opponent_rating=game['opponent_rating'], new_game=key in added,
            training_eligible=key in training,
            features=dict(zip(FEATURES, features[key].tolist())),
            errors100=sum(row['scores']['played']['cp_loss'] >= 100 for row in rows),
            errors300=sum(row['scores']['played']['cp_loss'] >= 300 for row in rows),
            recoverable_large_point_drops=len(errors),
            first_large_drop=None if first is None else dict(ply=first['ply'], **{
                name: first['scores']['played'][name] for name in ('move', 'point_loss', 'cp_loss')}),
            quality_estimate=predictions.get(key, (None, None))[0],
            rating_only_estimate=predictions.get(key, (None, None))[1]))
    pool = [row for row in games if row['rated'] and row['perf'] == 'blitz' and row['rating_diff'] is not None]
    return dict(games=len(games), decisions=sum(map(len, roots.values())),
        outcomes=dict(Counter('win' if row['score'] == 1 else 'loss' if row['score'] == 0 else 'draw' for row in games)),
        new_game_ids=added, new_rated_games=len(eligible),
        new_games_excluded_from_rating_score=sorted(set(added) - set(eligible)),
        new_game_score=float(y.mean()), rating_only_squared_error=float(np.mean((reference - y) ** 2)),
        quality_squared_error=float(np.mean((estimated - y) ** 2)), mean_error_reduction=float(gain.mean()),
        opponent_cluster_95=cluster_interval(gain, [records[key]['opponent'] for key in eligible]),
        independent_new_opponents=len({records[key]['opponent'] for key in eligible}),
        calibration_training_games=len(training), calibration_refitted_on_new_games=False,
        target_pool_reference={str(rating): float(sigmoid((rating - np.array([
            row['opponent_rating'] for row in pool])) * np.log(10) / 400).mean()) for rating in (2700, 2800)},
        target_pool_reference_note='Descriptive 400-point logistic expected score against recorded opponent ratings; not official Glicko2 or an offline Elo conversion.',
        game_diagnostics=details, offline_elo_conversion_established=False,
        target_2800_offline_threshold=None,
        training_allowed=False, neural_forwards=0, optimizer_updates=0, target_achieved=False,
        scope='Completed-game quality scored with unchanged earlier-game calibration. All new games in the snapshot are retained. '
              'Opponent bootstrap is descriptive, especially for small batches. This is not a pregame forecast, '
              'proof of alternate model rankings, or an official Elo. New games remain evaluation-only.')


def run(snapshot, registered, calibration_path, quality, output):
    paths = [snapshot / name for name in ('games.jsonl', 'positions.jsonl')]
    paths += [registered / 'games.jsonl', calibration_path, Path(__file__),
              Path(__file__).with_name('calibrate_lichess_history.py'),
              Path(__file__).with_name('lichess_quality.py')]
    frozen = {str(path.resolve()): artifact_info(path) for path in paths}
    calibration = json.loads(calibration_path.read_text())
    base = Path(calibration.get('input_root', calibration_path.resolve().parent.parent))
    for name, expected in calibration['inputs'].items():
        path = Path(name) if Path(name).is_absolute() else base / name
        if artifact_info(path) != expected:
            raise ValueError('Frozen calibration evidence changed: ' + str(path))
        frozen[str(path.resolve())] = expected
    if not any(Path(name).name == 'games.jsonl' and value == artifact_info(registered / 'games.jsonl')
               for name, value in calibration['inputs'].items()):
        raise ValueError('Calibration belongs to another registered snapshot')
    all_positions = list(map(json.loads, (snapshot / 'positions.jsonl').open()))
    positions = {(row['game_id'], row['ply']): row for row in all_positions}
    if len(positions) != len(all_positions):
        raise ValueError('Duplicate public game decision')
    roots, seen, teacher = {}, set(), None
    for folder in quality:
        report = json.loads((folder / 'review.json').read_text())
        source = Path(report['positions'])
        source = source if source.is_absolute() else folder.parent.parent / source
        if artifact_info(source) != report['protocol']['positions']:
            raise ValueError('Quality observations belong to another position snapshot')
        if report['protocol']['producer'] != artifact_info(Path(__file__).with_name('lichess_quality.py')):
            raise ValueError('Quality producer differs from the reviewed scorer')
        policy = {key: value for key, value in report['protocol'].items() if key != 'positions'}
        if teacher is not None and teacher != policy:
            raise ValueError('Teacher protocols must match exactly')
        teacher = policy
        frozen[str(folder.resolve())], frozen[str(source.resolve())] = artifact_info(folder), artifact_info(source)
        members = list(map(json.loads, source.open()))
        if len(members) != report['roots']:
            raise ValueError('Quality root count differs from its source')
        for row in members:
            key = row['game_id'], row['ply']
            if key in seen or positions.get(key) != row:
                raise ValueError('Missing, repeated or changed full game history')
            seen.add(key)
            if row['game_id'] not in roots:
                roots[row['game_id']] = json.loads((folder / (row['game_id'] + '.json')).read_text())
            observed = next((item for item in roots[row['game_id']] if item['ply'] == row['ply']), None)
            if observed is None or observed['scores']['played']['move'] != row['played_move']:
                raise ValueError('Quality action differs from the real played move')
            played = observed['scores']['played']
            if any(played[name] != value for name, value in loss(observed['best'], played['observed']).items()):
                raise ValueError('Quality loss differs from its teacher observations')
    if seen != set(positions):
        raise ValueError('Quality must cover every public game decision')
    result = compare(list(map(json.loads, (snapshot / 'games.jsonl').open())),
                     list(map(json.loads, (registered / 'games.jsonl').open())), roots, calibration)
    if any(artifact_info(path) != expected for path, expected in frozen.items()):
        raise ValueError('Scoring evidence changed during review')
    result['inputs'] = frozen
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('inputs', 'game_diagnostics', 'scope')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'registered', 'calibration', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--quality', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    run(args.snapshot, args.registered, args.calibration, args.quality, args.out)
