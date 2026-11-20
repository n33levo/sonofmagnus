import chess
import pytest

from scripts.hard_policy import harden


@pytest.mark.parametrize('policy, expected', [({'e2e4': .6, 'd2d4': .4}, {'e2e4': 1.}),
                                           ({'e2e4': .4, 'd2d4': .4, 'g1f3': .2}, {'e2e4': .5, 'd2d4': .5})])
def test_best_teacher_moves_keep_ties_q_targets_and_actual_history(policy, expected):
    row = {'fen': chess.STARTING_FEN, 'root_fen': chess.STARTING_FEN, 'history': [],
           'move_probs': policy, 'q_values': {'e2e4': .7, 'd2d4': .6}, 'weight': 3.}
    changed = harden(row)
    assert changed['move_probs'] == expected and row['move_probs'] == policy
    assert {key: value for key, value in changed.items() if key != 'move_probs'} == {
        key: value for key, value in row.items() if key != 'move_probs'}


@pytest.mark.parametrize('policy', [{}, {'e2e4': float('nan')}, {'e2e4': .6}, {'e2e5': 1.}, {'e2e4': 1.1, 'd2d4': -.1}])
def test_bad_teacher_targets_are_rejected(policy):
    with pytest.raises(ValueError):
        harden({'fen': chess.STARTING_FEN, 'move_probs': policy})


def test_single_move_expert_schema_keeps_the_original_move_and_q_label():
    row = {'fen': chess.STARTING_FEN, 'move': 'e2e4', 'q_values': {'e2e4': .8}}
    converted = harden(row)
    assert converted == {**row, 'move_probs': {'e2e4': 1.}}
    assert 'move_probs' not in row
