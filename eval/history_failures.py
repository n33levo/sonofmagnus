"""Locate costly decisions in complete recorded games, preserving the frozen split."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from eval.history_validation import validate_split
from model.artifacts import artifact_info


def summarize(game, roots, split):
    plies = [row['ply'] for row in roots]
    if (not roots or len(roots) != game['own_moves'] or plies != sorted(set(plies))
            or any(row['game_id'] != game['game_id'] for row in roots)):
        raise ValueError('Require every recorded decision once, in game order')
    failures = []
    for row in roots:
        best, played = row['best'], row['scores']['played']
        values = [best['points'], played['point_loss'], played['cp_loss']]
        if not np.isfinite(values).all() or not 0 <= best['points'] <= 1 or played['point_loss'] < 0:
            raise ValueError('Require finite external action observations')
        # Ignore mistakes after the evaluator already considers recovery unlikely.
        if best['points'] >= .25 and played['point_loss'] >= .1 and played['cp_loss'] > 40:
            failures.append(dict(ply=row['ply'], move=played['move'], alternative=best['move'],
                point_loss=played['point_loss'], cp_loss=played['cp_loss'], best_points=best['points'],
                phase='opening' if row['ply'] < 20 else 'endgame' if row['pieces'] <= 10 else 'middlegame'))
    return dict(game_id=game['game_id'], url=game['url'], score=game['score'],
        opponent=game['opponent'], opponent_rating=game['opponent_rating'], color=game['color'],
        status=game['status'], split=split, decisions=len(roots), costly_decisions=len(failures),
        first_costly_decision=failures[0] if failures else None,
        mean_point_loss=float(np.mean([row['scores']['played']['point_loss'] for row in roots])))


def run(folder, quality, calibration_path, output):
    if output.exists():
        raise ValueError('Use a new report path')
    paths = [folder / 'games.jsonl', folder / 'positions.jsonl', quality, calibration_path, Path(__file__),
             Path(__file__).with_name('history_validation.py')]
    inputs = {str(path.resolve()): artifact_info(path) for path in paths}
    calibration = json.loads(calibration_path.read_text())
    review = json.loads((quality / 'review.json').read_text())
    if (review['protocol']['positions'] != artifact_info(folder / 'positions.jsonl')
            or review['protocol']['nodes'] != 100000):
        raise ValueError('Quality observations belong to another history snapshot')
    if not any(Path(key).name == 'games.jsonl' and value == artifact_info(folder / 'games.jsonl')
               for key, value in calibration['inputs'].items()):
        raise ValueError('Calibration belongs to another game snapshot')
    games = list(map(json.loads, (folder / 'games.jsonl').open()))
    train, test = validate_split(games, calibration['chronological_validation'])
    earlier, later = {row['game_id'] for row in train}, {row['game_id'] for row in test}
    details = [summarize(game, json.loads((quality / (game['game_id'] + '.json')).read_text()),
        'earlier-training' if game['game_id'] in earlier else
        'later-development' if game['game_id'] in later else 'missing-rating-delta') for game in games]
    if review['games'] != len(details) or review['roots'] != sum(row['decisions'] for row in details):
        raise ValueError('Quality review must cover every game and decision')
    outcomes = {}
    for name, score in [('win', 1.), ('draw', .5), ('loss', 0.)]:
        rows = [row for row in details if row['score'] == score]
        outcomes[name] = dict(games=len(rows), with_costly_decision=sum(row['costly_decisions'] > 0 for row in rows),
            first_error_phase=dict(Counter(row['first_costly_decision']['phase'] for row in rows
                                          if row['first_costly_decision'])))
    result = dict(games=len(details), decisions=review['roots'], actual_score=float(np.mean([row['score'] for row in details])),
        outcomes=outcomes, earlier_games=len(earlier), later_games=len(later), details=details,
        definition='First own decision losing at least0.1 expected points and more than40cp, '
            'when the independent best action retains at least0.25 expected points. Fixed100000-node observations.',
        scope='Diagnostic on actual mixed-model games, including wins and draws. A better alternate move does not '
            'establish a counterfactual win. Earlier-game membership alone does not admit a training row; '
            'all existing exclusions and fresh source-action checks still apply. Later games stay out of training.',
        inputs=inputs, new_neural_forwards=0, automatic_training=False, automatic_rated=False,
        offline_elo_conversion_established=False)
    if any(artifact_info(path) != identity for path, identity in inputs.items()):
        raise ValueError('Diagnostic inputs changed')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('games', 'decisions', 'actual_score', 'outcomes')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=Path, default=Path('data/lichess-history'))
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.games, args.quality, args.calibration, args.out)
