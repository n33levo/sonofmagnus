import copy

import pytest

from eval.history_failures import summarize


def example():
    game = dict(game_id='a', url='https://lichess.org/a', score=0., opponent='opponent',
        opponent_rating=1900, color='black', status='mate', own_moves=3)
    roots = [dict(game_id='a', ply=ply, pieces=pieces, best=dict(points=points, move='e7e5'),
        scores=dict(played=dict(point_loss=loss, cp_loss=cp, move='c7c5')))
        for ply, pieces, points, loss, cp in [(1, 32, .01, .01, 900),
            (21, 20, .5, .2, 200), (71, 6, .8, .5, 500)]]
    return game, roots


def test_first_recoverable_error_precedes_worst_error_and_ignores_doomed_positions():
    game, roots = example()
    result = summarize(game, roots, 'later-development')
    assert result['first_costly_decision']['ply'] == 21
    assert result['first_costly_decision']['phase'] == 'middlegame'
    assert result['costly_decisions'] == 2 and result['split'] == 'later-development'
    assert result['score'] == 0.


@pytest.mark.parametrize('score', [0., .5, 1.])
def test_keeps_mistakes_in_wins_and_draws_without_rewriting_real_outcome(score):
    game, roots = example()
    game['score'] = score
    result = summarize(game, roots, 'earlier-training')
    assert result['score'] == score and result['costly_decisions'] == 2


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'out-of-order', 'wrong-game', 'nonfinite'])
def test_rejects_incomplete_or_corrupted_game_analysis(corruption):
    game, roots = example()
    roots = copy.deepcopy(roots)
    if corruption == 'missing':roots.pop()
    elif corruption == 'duplicate':roots[1] = roots[0]
    elif corruption == 'out-of-order':roots.reverse()
    elif corruption == 'wrong-game':roots[0]['game_id'] = 'b'
    else:roots[0]['scores']['played']['point_loss'] = float('nan')
    with pytest.raises(ValueError):
        summarize(game, roots, 'earlier-training')
