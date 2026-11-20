"""Audit the frozen history score against real outcomes and new opponents."""
import argparse
import json
from pathlib import Path

import numpy as np

from eval.calibrate_lichess_history import FEATURES, cluster_interval, fit, game_features, predict
from model.artifacts import artifact_info


def validate_split(games, split):
    records = {row['game_id']: row for row in games}
    train, test = split['training_game_ids'], split['test_game_ids']
    cohort = {row['game_id'] for row in games if row['rated'] and row['perf'] == 'blitz'
              and row['rating_diff'] is not None}
    if (len(records) != len(games) or not train or not test or
            len(set(train + test)) != len(train) + len(test) or set(train + test) != cohort):
        raise ValueError('Require a complete, distinct rated-blitz calibration split')
    if max(records[key]['finished_ms'] for key in train) >= min(records[key]['created_ms'] for key in test):
        raise ValueError('Training outcomes must precede every validation game')
    return [records[key] for key in train], [records[key] for key in test]


def run(folder, quality, calibration_path, output):
    paths = [folder / 'games.jsonl', folder / 'positions.jsonl', quality, calibration_path, Path(__file__),
             Path(__file__).with_name('calibrate_lichess_history.py')]
    frozen = {str(path.resolve()): artifact_info(path) for path in paths}
    calibration = json.loads(calibration_path.read_text())
    input_root = Path(calibration.get('input_root', calibration_path.resolve().parent.parent))
    for path, expected in calibration['inputs'].items():
        resolved = Path(path) if Path(path).is_absolute() else input_root / path
        if artifact_info(resolved) != expected:
            raise ValueError('Frozen calibration input changed: ' + path)
    for path in (folder / 'games.jsonl', folder / 'positions.jsonl', quality / 'review.json'):
        if not any(Path(key).name == path.name and info == artifact_info(path)
                   for key, info in calibration['inputs'].items()):
            raise ValueError('Validation belongs to another history snapshot')
    games = list(map(json.loads, (folder / 'games.jsonl').open()))
    train, test = validate_split(games, calibration['chronological_validation'])
    roots = {row['game_id']: json.loads((quality / (row['game_id'] + '.json')).read_text()) for row in games}
    if any(len(roots[row['game_id']]) != row['own_moves'] or
           any(root['game_id'] != row['game_id'] for root in roots[row['game_id']]) for row in games):
        raise ValueError('Quality data must cover every actual game decision')
    features = {key: game_features(rows) for key, rows in roots.items()}
    saved = {row['game_id']: row for row in calibration['game_features']}
    if len(saved) != len(train) + len(test) or set(saved) != {row['game_id'] for row in train + test}:
        raise ValueError('Stored calibration features do not cover its cohort')
    for row in train + test:
        values = saved[row['game_id']]
        if (row['score'] != values['score'] or row['opponent'] != values['opponent'] or
                row['opponent_rating'] != values['opponent_rating'] or
                not np.array_equal(features[row['game_id']], [values['features'][key] for key in FEATURES])):
            raise ValueError('Stored features or outcomes differ from the verified games')

    def arrays(rows):
        return (np.stack([features[row['game_id']] for row in rows]),
                np.array([row['score'] for row in rows]),
                np.array([row['own_rating'] - row['opponent_rating'] for row in rows]) * np.log(10) / 400)

    x, y, offset = arrays(train)
    model = {key: np.array(value) for key, value in calibration['fitted_model'].items()}
    rebuilt = fit(x, y, offset)
    if any(not np.allclose(model[key], value, rtol=0, atol=1e-12) for key, value in rebuilt.items()):
        raise ValueError('Frozen model cannot be reproduced from earlier games')
    baseline = fit(np.zeros((len(train), 0)), y, offset)
    x, y, offset = arrays(test)
    estimated = predict(model, x, offset)
    reference = predict(baseline, np.zeros((len(test), 0)), offset)
    gain = (reference - y) ** 2 - (estimated - y) ** 2
    opponents = {row['opponent'] for row in train}
    unseen = np.array([row['opponent'] not in opponents for row in test])
    cohorts = {}
    for name, mask in [('all_later', np.ones(len(test), dtype=bool)),
                       ('seen_opponents', ~unseen), ('unseen_opponents', unseen)]:
        if not mask.any():
            cohorts[name] = dict(games=0, supported=False)
            continue
        interval = cluster_interval(gain[mask], [row['opponent'] for row, keep in zip(test, mask) if keep])
        cohorts[name] = dict(games=int(mask.sum()), rating_only_squared_error=float(np.mean((reference[mask] - y[mask]) ** 2)),
            quality_squared_error=float(np.mean((estimated[mask] - y[mask]) ** 2)),
            error_reduction_95=interval, supported=interval[0] > 0)
    if any(artifact_info(path) != info for path, info in frozen.items()):
        raise ValueError('Validation inputs changed during review')
    result = dict(games=len(games), decisions=sum(map(len, roots.values())), training_games=len(train),
        later_games=len(test), cohorts=cohorts, fitted_quality_weights=model['weights'].tolist(),
        quality_penalties_all_nonpositive=bool((model['weights'][1:] <= 0).all()),
        zero_error_index_ceiling=float(100 * predict(model, np.zeros_like(x), offset).mean()),
        game_diagnostics=[dict(game_id=row['game_id'], url=row['url'], actual_score=row['score'],
            opponent=row['opponent'], opponent_rating=row['opponent_rating'],
            split='earlier-fitting' if row in train else 'later-validation' if row in test else 'missing-rating-delta',
            features=dict(zip(FEATURES, features[row['game_id']].tolist()))) for row in games],
        interpretation='Squared error uses actual 0/0.5/1 game scores, including draws. Features require completed games; '
            'this is retrospective association, not a pregame forecast or a causal test of changed models. '
            'The opponent slices audit the existing later development set; they are not fresh confirmation games.',
        selection='Beat the fixed incumbent on paired later-history point loss with its interval above zero, '
            'without new recoverable300cp errors; then confirm in complete external games and fresh rated play.',
        offline_elo_conversion_established=False, neural_training=False, automatic_rated=False,
        target_achieved=False, inputs=frozen)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('games', 'decisions', 'training_games', 'later_games', 'cohorts',
        'quality_penalties_all_nonpositive', 'zero_error_index_ceiling')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=Path, default=Path('data/lichess-history'))
    parser.add_argument('--quality', type=Path, default=Path('out/history-played-100k'))
    parser.add_argument('--calibration', type=Path, default=Path('out/history-recorded-calibration.json'))
    parser.add_argument('--out', type=Path, default=Path('out/history-outcome-validation.json'))
    args = parser.parse_args()
    run(args.games, args.quality, args.calibration, args.out)
