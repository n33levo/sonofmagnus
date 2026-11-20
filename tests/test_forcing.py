import chess
import pytest

from chessio.records import record_board
from train.forcing import branches, game_ids, position_key


def example(identifier='abcdEF12'):
    board = chess.Board()
    moves = 'e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8'.split()
    for move in moves:
        board.push_uci(move)
    return dict(game_id=identifier,fen=board.fen(),root_fen=chess.STARTING_FEN,history=moves,
        move='a2a3',parent_record=dict(file='admitted.jsonl',line=1),origin_ratings=[2800,2900])


def test_branches_preserve_the_actual_prefix_and_add_one_legal_move():
    parent = example()
    before = list(parent['history'])
    rows, summary = branches([parent],set(),set(),2,2026101913)
    assert len(rows)==2 and summary['source-games']==1
    for row in rows:
        board = record_board(row)
        assert board.is_valid() and not board.is_game_over(claim_draw=True)
        assert row['history'][:-1]==before and row['history'][-1]==row['branch_move']
        assert row['branch_move']!=parent['move'] and row['origin_game_id']==parent['game_id']
        assert not row['policy_supervision'] and row['actual_gpu_actor_review_pending']
    assert parent['history']==before


def test_whole_games_and_reflected_protected_roots_are_excluded():
    parent = example()
    assert branches([parent],set(),{'abcdEF12'},2,3)[0]==[]
    rows,_ = branches([parent],set(),set(),2,3)
    blocked = {position_key(record_board(row).mirror()) for row in rows}
    filtered,_ = branches([parent],blocked,set(),10,3)
    assert all(not ({position_key(record_board(row)),position_key(record_board(row).mirror())}&blocked) for row in filtered)


def test_sampling_is_deterministic_and_does_not_duplicate_contexts():
    parent = example()
    first,_ = branches([parent,dict(parent,game_id='otherGM1')],set(),set(),8,7)
    second,_ = branches([parent,dict(parent,game_id='otherGM1')],set(),set(),8,7)
    assert first==second and len({row['game_id'] for row in first})==len(first)
    assert all(sum(item['origin_game_id']==row['origin_game_id'] for item in first)<=2 for row in first)


def test_game_aliases_do_not_bypass_source_exclusions():
    assert 'abcdEF12' in game_ids(dict(game_id='external-abcdEF12-0'))
    assert 'abcdEF12' in game_ids(dict(game_url='https://lichess.org/abcdEF12WXYZ#12'))
    parent=example();parent['source_game']='external-denyGM12-1'
    assert branches([parent],set(),{'denyGM12'},2,7)[0]==[]


@pytest.mark.parametrize('limit,cap', [(0,2),(-1,2),(True,2),(2,0),(2,False)])
def test_invalid_limits_cannot_silently_change_the_population(limit,cap):
    with pytest.raises(ValueError):
        branches([example()],set(),set(),limit,7,cap)
