"""Join saved online model reviews to the complete public game snapshot."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from model.artifacts import artifact_info


def assign(games, reviews, account):
    records = {row['game_id']: row for row in games}
    if len(records) != len(games):
        raise ValueError('Duplicate public game IDs')
    claims, rejected = defaultdict(dict), []
    for path, review in reviews:
        rows = review.get('games_reviewed')
        if review.get('account') != account or not isinstance(rows, list):
            continue
        reason = None
        model = review.get('model', {})
        if (not review.get('source_hashes_verified') or review.get('unmatched_decisions') != 0
                or len(model.get('sha256', '')) != 64 or model.get('model_bytes', 0) <= 0):
            reason = 'Incomplete source or neural-decision review'
        elif any(not {'id', 'score', 'opponent', 'clock', 'status', 'neural_moves', 'rating_updated'} <= row.keys()
                 for row in rows):
            reason = 'Older review lacks complete game metadata'
        elif (review.get('games') != len(rows) or len({row['id'] for row in rows}) != len(rows)
                or review.get('neural_moves_verified') != sum(row['neural_moves'] for row in rows)):
            reason = 'Incomplete or duplicated reviewed games'
        else:
            for row in rows:
                game = records.get(row['id'])
                if (game is None or any(row[left] != game[right] for left, right in
                        [('score', 'score'), ('opponent', 'opponent'), ('clock', 'clock'),
                         ('status', 'status'), ('neural_moves', 'own_moves')])
                        or row['rating_updated'] != (game['rating_diff'] is not None)):
                    reason = 'Review disagrees with the public snapshot: ' + row['id']
                    break
        if reason:
            rejected.append(dict(path=path, reason=reason))
            continue
        for row in rows:
            claims[row['id']].setdefault(model['sha256'], []).append(path)
    details = []
    for game in games:
        candidates = claims[game['game_id']]
        details.append(dict(game_id=game['game_id'], score=game['score'],
            opponent=game['opponent'], opponent_rating=game['opponent_rating'],
            model_sha256=next(iter(candidates)) if len(candidates) == 1 else None,
            attribution='saved-review' if len(candidates) == 1 else 'conflict' if candidates else 'unknown',
            claims=candidates))
    models = []
    for sha in sorted({row['model_sha256'] for row in details} - {None}):
        rows = [row for row in details if row['model_sha256'] == sha]
        outcomes = Counter({1.: 0, .5: 0, 0.: 0})
        outcomes.update(row['score'] for row in rows)
        models.append(dict(sha256=sha, games=len(rows), wins=outcomes[1.], draws=outcomes[.5],
            losses=outcomes[0.], score=sum(row['score'] for row in rows) / len(rows),
            mean_opponent_rating=sum(row['opponent_rating'] for row in rows) / len(rows)))
    return dict(games=len(games), attribution_counts=dict(Counter(row['attribution'] for row in details)),
        models=models, rejected_reviews=rejected, game_details=details,
        automatic_rated=False, offline_elo_conversion_established=False,
        scope='Model identities come from saved independent online reviews, not fresh weight or runtime '
              'verification. Duplicate copies do not add games. Conflicting assignments remain unassigned. '
              'Different opponent pools and dates make these descriptive results unsuitable for ranking '
              'models or assigning the current account rating to a checkpoint.')


def run(games_path, review_paths, output, account):
    paths = list(dict.fromkeys([games_path, *review_paths, Path(__file__).resolve()]))
    frozen = {str(path.resolve()): artifact_info(path) for path in paths}
    reviews = [(str(path.resolve()), json.loads(path.read_text())) for path in review_paths]
    result = assign(list(map(json.loads, games_path.open())), reviews, account)
    if any(artifact_info(path) != identity for path, identity in frozen.items()):
        raise ValueError('Attribution inputs changed during review')
    result['inputs'] = frozen
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('games', 'attribution_counts', 'models', 'rejected_reviews')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--account', required=True)
    args = parser.parse_args()
    run(args.games, args.reviews, args.out, args.account)
