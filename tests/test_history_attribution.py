from copy import deepcopy

import pytest

from eval.history_attribution import assign


def observations():
    games = [dict(game_id='a', score=0., opponent='bot', opponent_rating=2400,
                  clock=dict(initial=180, increment=2), status='mate', own_moves=30, rating_diff=-5),
             dict(game_id='b', score=.5, opponent='other', opponent_rating=2600,
                  clock=dict(initial=180, increment=2), status='draw', own_moves=40, rating_diff=None)]
    reviewed = [dict(id=row['game_id'], score=row['score'], opponent=row['opponent'],
        clock=row['clock'], status=row['status'], neural_moves=row['own_moves'],
        rating_updated=row['rating_diff'] is not None) for row in games]
    review = dict(account='example-bot', games=2, games_reviewed=reviewed,
        source_hashes_verified=True, unmatched_decisions=0, neural_moves_verified=70,
        model=dict(sha256='a' * 64, model_bytes=100))
    return games, review


def test_duplicate_reviews_do_not_duplicate_games_or_drop_missing_deltas():
    games, review = observations()
    result = assign(games, [('first', review), ('copy', deepcopy(review))], 'example-bot')
    assert result['attribution_counts'] == {'saved-review': 2}
    assert result['models'][0]['games'] == 2
    assert result['models'][0]['score'] == .25
    assert result['models'][0]['draws'] == 1 and result['models'][0]['losses'] == 1
    assert not result['offline_elo_conversion_established']


def test_conflicting_checkpoints_remain_unassigned():
    games, review = observations()
    other = deepcopy(review)
    other['model']['sha256'] = 'b' * 64
    result = assign(games, [('first', review), ('other', other)], 'example-bot')
    assert result['attribution_counts'] == {'conflict': 2}
    assert result['models'] == []


@pytest.mark.parametrize('change', ['wrong-result', 'missing-move', 'unmatched', 'stale-source', 'duplicate', 'old-metadata'])
def test_incomplete_or_inconsistent_reviews_cannot_assign_models(change):
    games, review = observations()
    if change == 'wrong-result':review['games_reviewed'][0]['score'] = 1.
    elif change == 'missing-move':review['games_reviewed'][0]['neural_moves'] -= 1
    elif change == 'unmatched':review['unmatched_decisions'] = 1
    elif change == 'stale-source':review['source_hashes_verified'] = False
    elif change == 'duplicate':review['games_reviewed'][1] = deepcopy(review['games_reviewed'][0])
    else:del review['games_reviewed'][0]['rating_updated']
    result = assign(games, [('broken', review)], 'example-bot')
    assert result['attribution_counts'] == {'unknown': 2}
    assert len(result['rejected_reviews']) == 1


def test_duplicate_public_games_are_rejected():
    games, review = observations()
    with pytest.raises(ValueError, match='Duplicate'):
        assign(games + [games[0]], [('review', review)], 'example-bot')
