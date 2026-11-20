import pytest

from scripts.lichess_history import replay


def record():
    return dict(id='example1', variant='standard', status='mate', rated=True, perf='blitz',
        createdAt=1, lastMoveAt=2, winner='black',
        players={'white': {'user': {'id': 'example-bot'}, 'rating': 2300, 'ratingDiff': -5},
                 'black': {'user': {'id': 'opponent'}, 'rating': 2400}},
        moves='f3 e5 g4 Qh4#', clocks=[18000] * 4,
        pgn='[Result "0-1"]\n\n1. f3 e5 2. g4 Qh4# 0-1')


def test_server_pgn_and_san_replay_preserves_the_actual_side_and_history():
    game, roots = replay(record(), 'example-bot')
    assert game['score'] == 0 and game['own_moves'] == 2 and game['rating_diff'] == -5
    assert roots[0]['played_move'] == 'f2f3' and roots[0]['history'] == []
    assert roots[1]['history'] == ['f2f3', 'e7e5']
    assert roots[1]['played_move'] == 'g2g4'


def test_black_side_and_a_missing_rating_change_are_retained():
    row = record()
    row['players']['white']['user']['id'] = 'opponent'
    row['players']['black']['user']['id'] = 'example-bot'
    game, roots = replay(row, 'example-bot')
    assert game['score'] == 1 and game['rating_diff'] is None
    assert roots[0]['played_move'] == 'e7e5' and roots[0]['history'] == ['f2f3']


def test_an_extra_terminal_clock_is_recorded_without_shifting_move_clocks():
    row = record()
    row['clocks'].append(0)
    game, roots = replay(row, 'example-bot')
    assert game['extra_terminal_clock']
    assert [root['clock_after_cs'] for root in roots] == [18000, 18000]


@pytest.mark.parametrize('mutation', ['result', 'moves', 'clocks'])
def test_inconsistent_server_records_fail(mutation):
    row = record()
    if mutation == 'result':row['winner'] = 'white'
    elif mutation == 'moves':row['moves'] = 'e4 e5 g4 Qh4#'
    else:row['clocks'] = [100]
    with pytest.raises(ValueError):replay(row, 'example-bot')
