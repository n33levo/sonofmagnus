"""Check whether historical move quality predicts held-out game results."""
import argparse
import json
from pathlib import Path

import numpy as np

from model.artifacts import artifact_info


FEATURES = ('mean_point_loss', 'max_point_loss', 'fraction_100cp', 'fraction_300cp')


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -np.asarray(x)))


def fit(x, y, offset, alpha=10.):
    mean = x.mean(0)
    scale = np.maximum(x.std(0), 1e-6)
    z = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    weights = np.zeros(z.shape[1])
    penalty = np.diag([0.] + [alpha] * x.shape[1])
    for _ in range(50):
        probability = sigmoid(offset + z @ weights)
        gradient = z.T @ (probability - y) + penalty @ weights
        hessian = z.T @ (z * (probability * (1 - probability))[:, None]) + penalty + np.eye(z.shape[1]) * 1e-9
        change = np.linalg.solve(hessian, gradient)
        weights -= change
        if np.max(np.abs(change)) < 1e-8:
            break
    if not np.isfinite(weights).all():
        raise ValueError('Non-finite retrospective calibration')
    return dict(weights=weights, mean=mean, scale=scale)


def predict(model, x, offset):
    z = np.column_stack((np.ones(len(x)), (x - model['mean']) / model['scale']))
    return sigmoid(offset + z @ model['weights'])


def cluster_interval(values, groups):
    unique = list(dict.fromkeys(groups))
    sums = np.array([values[np.array(groups) == group].sum() for group in unique])
    sizes = np.array([sum(item == group for item in groups) for group in unique])
    indices = np.random.default_rng(2026101811).integers(len(unique), size=(10000, len(unique)))
    return np.quantile(sums[indices].sum(1) / sizes[indices].sum(1), [.025, .975]).tolist()


def game_features(roots, name='played'):
    rows = [row['scores'][name] for row in roots]
    return np.array([np.mean([row['point_loss'] for row in rows]),
        max(row['point_loss'] for row in rows), np.mean([row['cp_loss'] >= 100 for row in rows]),
        np.mean([row['cp_loss'] >= 300 for row in rows])])


def performance_index(ratings, scores):
    low, high = 0., 4000.
    for _ in range(80):
        middle = (low + high) / 2
        expected = sigmoid((middle - ratings) * np.log(10) / 400).sum()
        if expected < scores.sum():low = middle
        else:high = middle
    return (low + high) / 2


def run(folder, quality, output):
    review = json.loads((quality / 'review.json').read_text())
    games_path = folder / 'games.jsonl'
    positions_path = folder / 'positions.jsonl'
    if artifact_info(positions_path) != review['protocol']['positions']:
        raise ValueError('Analysis belongs to another game snapshot')
    games = list(map(json.loads, games_path.open()))
    games.sort(key=lambda row: (row['created_ms'], row['game_id']))
    roots = {row['game_id']: json.loads((quality / (row['game_id'] + '.json')).read_text()) for row in games}
    assert len(games) == review['games'] and sum(map(len, roots.values())) == review['roots']
    assert all(len(roots[row['game_id']]) == row['own_moves'] for row in games)
    # Rating changes identify the same157-game cohort as the official perf counter.
    cohort = [row for row in games if row['rating_diff'] is not None and row['rated'] and row['perf'] == 'blitz']
    x = np.stack([game_features(roots[row['game_id']]) for row in cohort])
    y = np.array([row['score'] for row in cohort])
    offset = np.array([row['own_rating'] - row['opponent_rating'] for row in cohort]) * np.log(10) / 400
    cut = 2 * len(cohort) // 3
    training, test = np.arange(len(cohort)) < cut, np.arange(len(cohort)) >= cut
    model = fit(x[training], y[training], offset[training])
    baseline = fit(np.zeros((training.sum(), 0)), y[training], offset[training])
    predicted = predict(model, x[test], offset[test])
    reference = predict(baseline, np.zeros((test.sum(), 0)), offset[test])
    improvement = np.square(reference - y[test]) - np.square(predicted - y[test])
    opponents = [row['opponent'] for row, keep in zip(cohort, test) if keep]
    interval = cluster_interval(improvement, opponents)
    approved = float(improvement.mean()) > 0 and interval[0] > 0
    details = []
    for row, values in zip(cohort, x):
        details.append(dict(game_id=row['game_id'], score=row['score'], opponent=row['opponent'],
            opponent_rating=row['opponent_rating'], features=dict(zip(FEATURES, values.tolist()))))
    worst = []
    for game in games:
        for root in roots[game['game_id']]:
            value = root['scores']['played']
            worst.append(dict(game_id=game['game_id'], url=game['url'], ply=root['ply'],
                move=value['move'], point_loss=value['point_loss'], cp_loss=value['cp_loss'],
                actual_game_score=game['score']))
    worst.sort(key=lambda row: (row['point_loss'], row['cp_loss']), reverse=True)
    outcome_features = {label: dict(games=int((y == score).sum()),
        mean_features=dict(zip(FEATURES, x[y == score].mean(0).tolist())))
        for label, score in [('win', 1.), ('draw', .5), ('loss', 0.)]}
    opponent_ratings = np.array([row['opponent_rating'] for row in cohort])
    result = dict(games=len(games), rating_change_cohort=len(cohort), positions=review['roots'],
        actual_score_all=sum(row['score'] for row in games) / len(games), actual_score_cohort=float(y.mean()),
        historical_logistic_performance_index=performance_index(opponent_ratings, y),
        target_pool_expectations={str(rating): float(sigmoid((rating - opponent_ratings) * np.log(10) / 400).mean()) for rating in (2700, 2800)},
        rating_scale_note='Descriptive400-point logistic model of recorded opponent ratings and actual results; not Lichess Glicko2 or a prospective bot rating.',
        features=FEATURES, outcome_features=outcome_features,
        chronological_validation=dict(training_games=int(training.sum()), test_games=int(test.sum()),
            training_game_ids=[row['game_id'] for row, keep in zip(cohort, training) if keep],
            test_game_ids=[row['game_id'] for row, keep in zip(cohort, test) if keep],
            baseline_brier=float(np.mean(np.square(reference - y[test]))),
            quality_brier=float(np.mean(np.square(predicted - y[test]))),
            mean_brier_improvement=float(improvement.mean()), opponent_cluster_bootstrap_95=interval,
            baseline='Pregame rating difference with a training-only intercept correction',
            decision='retrospective-quality-signal-supported' if approved else 'no-supported-outcome-calibration',
            scope='Quality features are computed from completed games. This tests retrospective association on later game outcomes; it is not a prediction available at game start.'),
        fitted_model={key: value.tolist() for key, value in model.items()},
        worst_recorded_moves=worst[:30], game_features=details,
        input_root=str(Path.cwd().resolve()),
        inputs={str(path): artifact_info(path) for path in [games_path, positions_path, quality / 'review.json', Path(__file__)]},
        automatic_rated=False, neural_training=False, target_achieved=False,
        scope='Retrospective mixed-model history. No held-out outcome enters fitting, scaling or feature selection. '
              'Even supported association cannot turn a counterfactual root choice into a verified game result or2800 rating.')
    if 'candidate' in review['summary']:
        differences = np.array([game_features(roots[row['game_id']])[0] - game_features(roots[row['game_id']], 'candidate')[0] for row in cohort])
        result['candidate_vs_historical_played'] = dict(mean_point_loss_reduction_per_game=float(differences.mean()),
            opponent_cluster_bootstrap_95=cluster_interval(differences, [row['opponent'] for row in cohort]),
            scope='Same recorded positions, alternate decisions. Actual historical moves used several models; this does not establish a gain over the current incumbent or new game Elo.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('games', 'positions', 'actual_score_cohort',
        'historical_logistic_performance_index', 'target_pool_expectations', 'chronological_validation')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=Path, default=Path('data/lichess-history'))
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.games, args.quality, args.out)
