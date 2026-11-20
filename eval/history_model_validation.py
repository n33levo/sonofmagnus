"""Check the recorded-game quality signal without fitting on the tested model."""
import argparse
import json
from pathlib import Path

import numpy as np

from eval.calibrate_lichess_history import FEATURES, cluster_interval, fit, predict
from model.artifacts import artifact_info


def compare(games, diagnostics, assignments):
    records = {row['game_id']: row for row in games}
    quality = {row['game_id']: row for row in diagnostics}
    identities = {row['game_id']: row for row in assignments}
    if (len(records) != len(games) or len(quality) != len(diagnostics)
            or len(identities) != len(assignments) or records.keys() != quality.keys()
            or records.keys() != identities.keys()):
        raise ValueError('Require every game exactly once in all three inputs')
    rows, excluded = [], []
    for key, game in records.items():
        observation, identity = quality[key], identities[key]
        if (any(identity[name] != game[name] for name in ('score', 'opponent', 'opponent_rating'))
                or observation['actual_score'] != game['score']):
            raise ValueError('Outcomes or opponents differ from the public games')
        split = observation['split']
        expected = game['rated'] and game['perf'] == 'blitz' and game['rating_diff'] is not None
        if split not in ('earlier-fitting', 'later-validation', 'missing-rating-delta') or expected != (split != 'missing-rating-delta'):
            raise ValueError('Invalid rated-blitz scoring cohort')
        if not expected or identity['attribution'] != 'saved-review':
            excluded.append(dict(game_id=key, reason='outside-rating-cohort' if not expected else 'unassigned-model'))
            continue
        sha = identity['model_sha256']
        values = [observation['features'][name] for name in FEATURES]
        if (not isinstance(sha, str) or len(sha) != 64 or game['score'] not in (0., .5, 1.)
                or not np.isfinite(values + [game['own_rating'], game['opponent_rating']]).all()):
            raise ValueError('Invalid model identity, features or score')
        rows.append(dict(game_id=key, model=sha, split=split, opponent=game['opponent'],
            created_ms=game['created_ms'], finished_ms=game['finished_ms'],
            features=values, score=game['score'],
            offset=(game['own_rating'] - game['opponent_rating']) * np.log(10) / 400))
    earlier = [row for row in rows if row['split'] == 'earlier-fitting']
    later = [row for row in rows if row['split'] == 'later-validation']
    if (not earlier or not later or len({row['model'] for row in earlier}) < 3
            or max(row['finished_ms'] for row in earlier) >= min(row['created_ms'] for row in later)):
        raise ValueError('Require three earlier models and a strictly later scoring cohort')
    folds, predictions = [], []
    for sha in sorted({row['model'] for row in rows}):
        train = [row for row in earlier if row['model'] != sha]
        test = [row for row in rows if row['model'] == sha]
        x = np.array([row['features'] for row in train])
        y = np.array([row['score'] for row in train])
        offset = np.array([row['offset'] for row in train])
        model = fit(x, y, offset)
        baseline = fit(np.zeros((len(train), 0)), y, offset)
        x = np.array([row['features'] for row in test])
        offset = np.array([row['offset'] for row in test])
        estimated = predict(model, x, offset)
        reference = predict(baseline, np.zeros((len(test), 0)), offset)
        folds.append(dict(model=sha, training_game_ids=[row['game_id'] for row in train],
            tested_game_ids=[row['game_id'] for row in test]))
        predictions.extend(dict(game_id=row['game_id'], model=sha, split=row['split'],
            opponent=row['opponent'], actual_score=row['score'],
            quality_estimate=float(value), rating_only_estimate=float(base))
            for row, value, base in zip(test, estimated, reference))
    cohorts = {}
    for split in ('earlier-fitting', 'later-validation'):
        selected = [row for row in predictions if row['split'] == split]
        y = np.array([row['actual_score'] for row in selected])
        quality_error = (np.array([row['quality_estimate'] for row in selected]) - y) ** 2
        rating_error = (np.array([row['rating_only_estimate'] for row in selected]) - y) ** 2
        gain = rating_error - quality_error
        models = [row['model'] for row in selected]
        cohorts[split] = dict(games=len(selected), models=len(set(models)),
            rating_only_squared_error=float(rating_error.mean()), quality_squared_error=float(quality_error.mean()),
            mean_error_reduction=float(gain.mean()),
            opponent_cluster_95=cluster_interval(gain, [row['opponent'] for row in selected]),
            model_cluster_95=cluster_interval(gain, models) if len(set(models)) > 1 else None)
    return dict(all_games=len(games), scored_games=len(rows), excluded_games=excluded,
        cohorts=cohorts, folds=folds, game_predictions=predictions,
        fitted_on_later_games=False, fitted_on_tested_checkpoint=False,
        offline_elo_conversion_established=False, target_2800_offline_threshold=None,
        automatic_rated=False, promoted=False,
        scope='Retrospective completed-game features, with fixed ridge strength and checkpoint exclusions. '
              'Earlier folds share training games; opponent and model intervals are separate descriptive resamples. '
              'A positive result is not a prospective checkpoint ranking, causal gain, or Lichess rating.')


def run(games, validation, attribution, output):
    paths = [games, validation, attribution, Path(__file__),
             Path(__file__).with_name('calibrate_lichess_history.py')]
    frozen = {str(path.resolve()): artifact_info(path) for path in paths}
    reports = [json.loads(path.read_text()) for path in (validation, attribution)]
    for report in reports:
        for path, expected in report['inputs'].items():
            if artifact_info(path) != expected:
                raise ValueError('Saved review input changed: ' + path)
            frozen[str(Path(path).resolve())] = expected
        if not any(Path(path).name == 'games.jsonl' and expected == artifact_info(games)
                   for path, expected in report['inputs'].items()):
            raise ValueError('Saved review belongs to another game snapshot')
    result = compare(list(map(json.loads, games.open())), reports[0]['game_diagnostics'], reports[1]['game_details'])
    if any(artifact_info(path) != expected for path, expected in frozen.items()):
        raise ValueError('Scoring inputs changed during review')
    result['inputs'] = frozen
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('all_games', 'scored_games', 'cohorts')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('games', 'validation', 'attribution', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.games, args.validation, args.attribution, args.out)
