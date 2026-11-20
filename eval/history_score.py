"""Compare models on the fixed later-game history benchmark."""
import argparse
import json
from pathlib import Path

import numpy as np

from eval.calibrate_lichess_history import cluster_interval, game_features, predict
from model.artifacts import artifact_info


def game_details(rows):
    worst = max(rows, key=lambda row: (row['scores']['candidate']['point_loss'],
                                      row['scores']['candidate']['cp_loss']))
    scores = [row['scores']['candidate'] for row in rows]
    return dict(roots=len(rows), mean_point_loss=float(np.mean([row['point_loss'] for row in scores])),
        errors100=sum(row['cp_loss'] >= 100 for row in scores), errors300=sum(row['cp_loss'] >= 300 for row in scores),
        worst_move=dict(ply=worst['ply'], **{key: worst['scores']['candidate'][key]
            for key in ('move', 'point_loss', 'cp_loss')}))


def run(games_path, calibration_path, baseline_path, candidate_path, output):
    inputs = {str(path.resolve()): artifact_info(path) for path in
        (games_path, calibration_path, baseline_path, candidate_path, Path(__file__),
         Path(__file__).with_name('calibrate_lichess_history.py'))}
    calibration = json.loads(calibration_path.read_text())
    if not any(Path(path).name == 'games.jsonl' and info == artifact_info(games_path)
               for path, info in calibration['inputs'].items()):
        raise ValueError('Calibration belongs to another game snapshot')
    reviews = [json.loads((path / 'review.json').read_text()) for path in (baseline_path, candidate_path)]
    if reviews[0]['protocol'] != reviews[1]['protocol']:
        raise ValueError('Compare identical histories and external scoring protocols')
    if any('candidate' not in review['summary'] for review in reviews):
        raise ValueError('Both folders must include verified model choices')
    records = list(map(json.loads, games_path.open()))
    all_games = {row['game_id']: row for row in records}
    if len(all_games) != len(records):
        raise ValueError('Duplicate game IDs')
    ids = calibration['chronological_validation']['test_game_ids']
    training_ids = calibration['chronological_validation']['training_game_ids']
    if len(set(ids)) != len(ids) or set(ids) & set(training_ids):
        raise ValueError('Later games must be distinct and excluded from calibration fitting')
    games = [all_games[game] for game in ids]
    features, roots, groups = [], [], []
    for path, review in zip((baseline_path, candidate_path), reviews):
        group = {game: json.loads((path / (game + '.json')).read_text()) for game in all_games}
        for game, rows in group.items():
            if (not rows or len(rows) != all_games[game]['own_moves'] or
                    any(row['game_id'] != game for row in rows) or
                    [row['ply'] for row in rows] != sorted({row['ply'] for row in rows})):
                raise ValueError('Incomplete, duplicated or mismatched game roots')
        if review['games'] != len(group) or review['roots'] != sum(map(len, group.values())):
            raise ValueError('Quality review does not cover every game')
        groups.append(group)
        features.append(np.stack([game_features(group[game], 'candidate') for game in ids]))
        roots.append({(row['game_id'], row['ply']): row for rows in group.values() for row in rows})
    if roots[0].keys() != roots[1].keys():
        raise ValueError('Model evaluation roots differ')
    models = {key: np.array(value) for key, value in calibration['fitted_model'].items()}
    offset = np.array([row['own_rating'] - row['opponent_rating'] for row in games]) * np.log(10) / 400
    supported = calibration['chronological_validation']['decision'] == 'retrospective-quality-signal-supported'
    indices = [float(100 * predict(models, values, offset).mean()) if supported else None for values in features]
    gains = features[0][:, 0] - features[1][:, 0]
    interval = cluster_interval(gains, [row['opponent'] for row in games])
    new_errors = {100: [], 300: []}
    for key, base in roots[0].items():
        candidate = roots[1][key]
        if base['best'] != candidate['best']:
            raise ValueError('Reference search observations differ')
        if key[0] not in ids:
            continue
        best = base['best']
        recoverable = (best.get('mate') or 0) > 0 or (best['cp'] is not None and best['cp'] >= -300)
        for threshold in new_errors:
            if recoverable and base['scores']['candidate']['cp_loss'] < threshold <= candidate['scores']['candidate']['cp_loss']:
                new_errors[threshold].append(dict(game_id=key[0], ply=key[1]))
    checks = dict(positive_paired_point_gain=float(gains.mean()) > 0 and interval[0] > 0,
        quality_index_improves=supported and indices[1] > indices[0], no_new_300cp_errors=not new_errors[300])
    details = []
    for game, record in all_games.items():
        details.append(dict(game_id=game, url='https://lichess.org/' + game, actual_score=record['score'],
            opponent=record['opponent'], opponent_rating=record['opponent_rating'],
            split='later-development' if game in ids else 'earlier-training' if game in
                training_ids else 'missing-rating-delta',
            baseline=game_details(groups[0][game]), candidate=game_details(groups[1][game])))
    if any(artifact_info(path) != info for path, info in inputs.items()):
        raise ValueError('History scoring inputs changed during review')
    result = dict(games=len(games), roots=sum(len(groups[0][game]) for game in ids),
        all_games=len(all_games), all_roots=len(roots[0]), game_diagnostics=details,
        baseline_quality_index=indices[0], candidate_quality_index=indices[1],
        mean_point_loss_reduction_per_game=float(gains.mean()), opponent_cluster_bootstrap_95=interval,
        new_100cp_errors=new_errors[100], new_300cp_errors=new_errors[300], checks=checks,
        decision='history-improvement-requires-external-games' if all(checks.values()) else 'no-qualified-history-improvement',
        scale='Higher0-100 index means better move quality under the fixed retrospective outcome model. It is not a prospective win percentage or an Elo conversion.',
        inputs=inputs,
        automatic_rated=False, promoted=False, target_achieved=False,
        scope='Fixed later-game development benchmark. Alternate root choices use recorded opponent histories, not newly played games. '
              'External full-game confirmation and official rated evidence remain required.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('games', 'roots', 'baseline_quality_index', 'candidate_quality_index',
        'mean_point_loss_reduction_per_game', 'opponent_cluster_bootstrap_95', 'decision')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=Path, default=Path('data/lichess-history/games.jsonl'))
    parser.add_argument('--calibration', type=Path, default=Path('out/history-recorded-calibration.json'))
    parser.add_argument('--baseline', type=Path, default=Path('out/history-incumbent-100k'))
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.games, args.calibration, args.baseline, args.candidate, args.out)
