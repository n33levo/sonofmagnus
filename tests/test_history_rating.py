import pytest

from scripts.check_history_rating import reconstruct


def observations():
    games = [dict(game_id='a', finished_ms=1, own_rating=2400, rating_diff=-10, rated=True, perf='blitz'),
             dict(game_id='b', finished_ms=2, own_rating=2390, rating_diff=5, rated=True, perf='blitz'),
             dict(game_id='missing', finished_ms=3, own_rating=2395, rating_diff=None, rated=True, perf='blitz')]
    account = dict(account='example-bot', title='BOT', utc='snapshot', blitz=dict(games=2, rating=2395, rd=45))
    return games, account


def test_exact_chain_keeps_missing_deltas_separate():
    games, account = observations()
    result = reconstruct(games[::-1], account, 'example-bot')
    assert result['reconstructed_rating'] == 2395 and result['recorded_updates'] == 2
    assert result['exact_transitions'] == 1 and result['missing_deltas'] == ['missing']
    assert not result['target_achieved']


def test_gap_cannot_be_hidden_by_a_matching_final_rating():
    games, account = observations()
    games[0]['rating_diff'] = -20
    with pytest.raises(ValueError, match='gap'):
        reconstruct(games, account, 'example-bot')


def test_stale_or_incomplete_snapshot_cannot_be_reconciled():
    games, account = observations()
    account['blitz']['games'] = 3
    with pytest.raises(ValueError, match='snapshot'):
        reconstruct(games, account, 'example-bot')
