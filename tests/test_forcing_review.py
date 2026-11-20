import copy
from collections import Counter
import json
from pathlib import Path

import pytest

from model.artifacts import artifact_info
from scripts.forcing_recovery import label_review


@pytest.fixture
def completed(tmp_path):
    rows=json.loads((Path(__file__).parent/'fixtures/anchor-labels.json').read_text())
    for row in rows:
        row['proposal_only']=True
    requests=[{k:row[k] for k in ('game_id','fen','root_fen','history','actor_choices',
        'screened_teacher_move','proposal_only','policy_supervision','actual_gpu_actor_review_pending')} for row in rows]
    for name,records in [('requests.jsonl',requests),('labels.jsonl',rows)]:
        (tmp_path/name).write_text(''.join(json.dumps(row)+'\n' for row in records))
    runtime=Path(__file__).resolve().parents[1]
    (tmp_path/'requests.json').write_text(json.dumps(dict(requested=len(rows),artifact=artifact_info(tmp_path/'requests.jsonl'),
        exclusions=[],label_runtime=str(runtime))))
    meta=dict(requested=len(rows),artifact=artifact_info(tmp_path/'labels.jsonl'),depth=22,wall_clock_cap=False,
        uci_options=dict(Threads=1,Hash=256,UCI_ShowWDL=True),exclusions=[],inputs={},
        counts=dict(Counter(row['alignment_status'] for row in rows)))
    (tmp_path/'labels.json').write_text(json.dumps(meta))
    return tmp_path,rows,meta


def rewrite(folder,rows,meta):
    path=folder/'labels.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    meta['artifact']=artifact_info(path)
    (folder/'labels.json').write_text(json.dumps(meta))


def test_actual_completed_teacher_records_remain_proposals(completed):
    folder,_,_=completed
    result=label_review(folder,{})
    assert result['complete'] and result['checked']==3 and result['fit_ready_targets']==0
    assert result['counts']=={'source-safe':1,'below-recovery':1,'confirmed-correction':1}
    assert result['actual_gpu_actor_review_pending'] and not result['automatic_training']


def test_missing_root_cannot_be_counted_complete(completed):
    folder,rows,meta=completed
    rows.pop();meta['counts']={'source-safe':1,'below-recovery':1,'incomplete-actions':1}
    rewrite(folder,rows,meta)
    result=label_review(folder,{})
    assert not result['complete'] and result['incomplete']==1 and result['checked']==2


@pytest.mark.parametrize('damage', ['duplicate','unjudged','trainable','wrong-history'])
def test_forged_completeness_or_scope_is_rejected(completed,damage):
    folder,rows,meta=completed
    if damage=='duplicate': rows[1]=copy.deepcopy(rows[0])
    elif damage=='unjudged': rows[2]['teacher_observations'][-1]['observed'].pop()
    elif damage=='trainable': rows[0]['policy_supervision']=True
    else: rows[0]['history']=rows[0]['history'][:-1]
    rewrite(folder,rows,meta)
    with pytest.raises((AssertionError,ValueError)):
        label_review(folder,{})
