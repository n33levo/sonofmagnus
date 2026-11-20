from copy import deepcopy

import chess
import pytest

from train.elite_labels import review_label


def record():
    observations = [dict(move=move, depth=22, cp=cp, raw=dict(cp=cp), wdl=wdl)
                    for move, cp, wdl in [('d2d4', 90, [300, 690, 10]), ('e2e4', -100, [10, 690, 300])]]
    return dict(fen=chess.STARTING_FEN, actor_choices=['e2e4'], played_grandmaster_move='d2d4',
        teacher_observations=[dict(kind='unrestricted', observed=observations[0]),
                              dict(kind='restricted-actions', requested=['d2d4', 'e2e4'], observed=observations)],
        centipawns=dict(d2d4=90, e2e4=-100), move='d2d4', alignment_status='confirmed-correction',
        actor_costs=dict(e2e4=190), teacher_wdl=[300, 690, 10], value=.29, complete_action_labels=False,
        reward_supervision=dict(labelled_moves=['d2d4', 'e2e4'], known_costs=[0, 190], expert_depth_min=22))


def test_complete_external_labels_are_admitted():
    assert review_label(record()) == 'confirmed-correction'


def test_bot_feedback_keeps_its_actual_source_instead_of_a_grandmaster_label():
    row = record()
    row['screened_teacher_move'] = row.pop('played_grandmaster_move')
    assert review_label(row) == 'confirmed-correction'


def test_missing_expert_is_rejected_even_when_every_actor_is_present():
    row = record()
    row['teacher_observations'][1]['observed'] = row['teacher_observations'][1]['observed'][1:]
    with pytest.raises(ValueError, match='Incomplete requested'):
        review_label(row)


@pytest.mark.parametrize('change', ['shallow', 'fake_wdl', 'cost', 'actor_cost', 'value', 'full_claim'])
def test_inconsistent_teacher_evidence_is_rejected(change):
    row = deepcopy(record())
    if change == 'shallow':row['teacher_observations'][1]['observed'][1]['depth'] = 21
    elif change == 'fake_wdl':row['teacher_observations'][1]['observed'][0]['wdl'] = [1, 2, 3]
    elif change == 'cost':row['reward_supervision']['known_costs'][1] = 191
    elif change == 'actor_cost':row['actor_costs']['e2e4'] = 191
    elif change == 'value':row['value'] = .9
    elif change == 'full_claim':row['complete_action_labels'] = True
    with pytest.raises(ValueError):review_label(row)
