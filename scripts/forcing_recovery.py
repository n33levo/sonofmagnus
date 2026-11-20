"""Prepare independent reply labels on legal branches from admitted strong games."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import chess.engine

from chessio.records import holdout_positions, record_board
from model.artifacts import artifact_info
from train.forcing import branches, game_ids, position_key
from train.elite_labels import review_label


def context(row):
    return row['fen'],row.get('root_fen',row['fen']),tuple(row.get('history',[]))


def prepare(source, output):
    protocol = source/'out/point-protocol.json'
    corpus = source/'out/point-corpus-review.json'
    plan, review = json.loads(protocol.read_text()),json.loads(corpus.read_text())
    data, originals = Path(plan['data']),source/'data/independent-source-inputs.jsonl'
    assert artifact_info(data)==plan['inputs'][str(data)]
    assert artifact_info(originals)==review['inputs'][str(originals)]
    exclusions = list(dict.fromkeys([*review['original_exclusions'],*review['exclusions']]))
    for path in exclusions:
        assert artifact_info(path)==plan['inputs'][path],path
    blocked, denied = holdout_positions(review['exclusions']),set()
    for path in review['original_exclusions']:
        for row in map(json.loads,Path(path).open()):
            denied.update(game_ids(row))
    ratings = {context(r):[r['played_rating'],r['opponent_rating']]
        for r in map(json.loads,originals.open())
        if min(r.get('played_rating',0),r.get('opponent_rating',0))>=2800}
    parents = []
    for line,row in enumerate(map(json.loads,data.open()),1):
        if context(row) in ratings and 16<=len(row.get('history',[]))<=160:
            parents.append(dict(row,parent_record=dict(file=str(data),line=line),origin_ratings=ratings[context(row)]))
    rows,counts = branches(parents,blocked,denied,4096,2026101913)
    if len(rows)!=4096:
        raise ValueError(f'Required 4096 branches, found {len(rows)}')
    selected = output/'branches.jsonl'
    with selected.open('x') as stream:
        for row in rows:
            print(json.dumps(row,separators=(',',':')),file=stream)
    return plan,exclusions,rows,counts,[protocol,corpus,data,originals,selected,*map(Path,exclusions)]


def label_review(output, frozen):
    requests,labels = output/'requests.jsonl',output/'labels.jsonl'
    parent = json.loads(requests.with_suffix('.json').read_text())
    requested = parent['requested']
    meta = json.loads(labels.with_suffix('.json').read_text())
    assert meta['artifact']==artifact_info(labels) and meta['requested']==requested
    assert meta['depth']==22 and not meta['wall_clock_cap'] and meta['uci_options']==dict(Threads=1,Hash=256,UCI_ShowWDL=True)
    assert all(artifact_info(p)==v for p,v in meta['inputs'].items())
    assert all(artifact_info(p)==v for p,v in frozen.items())
    blocked = holdout_positions(parent['exclusions'])
    expected = {context(r):r for r in map(json.loads,requests.open())}
    assert len(expected)==requested and meta['exclusions']==parent['exclusions']
    assert artifact_info(requests)==parent['artifact']
    counts,seen = Counter(),set()
    for row in map(json.loads,labels.open()):
        key = context(row)
        assert key in expected and key not in seen
        assert all(row[k]==v for k,v in expected[key].items() if k!='source')
        assert row['proposal_only'] and not row['policy_supervision'] and row['actual_gpu_actor_review_pending']
        board = record_board(row)
        assert not {position_key(board),position_key(board.mirror())}&blocked
        counts[review_label(row)]+=1
        seen.add(key)
    assert counts==Counter({k:v for k,v in meta['counts'].items() if not k.startswith('incomplete-')})
    assert sum(meta['counts'].values())==requested
    return dict(complete=len(seen)==requested,requested=requested,checked=len(seen),counts=dict(counts),
        incomplete=requested-len(seen),minimum_label_depth=22,wall_clock_cap=False,
        external_proposal='Weiss1.0 depth6, not a neural source capture',
        actual_gpu_actor_review_pending=True,fit_ready_targets=0,automatic_training=False,automatic_rated=False,
        scope='Independent supervision proposals only. Actor costs compare an external engine with Stockfish; they do not measure our model.',
        inputs={**frozen,**meta['inputs'],str(labels):artifact_info(labels),str(labels.with_suffix('.json')):artifact_info(labels.with_suffix('.json'))})


def run(source,output):
    output.mkdir(parents=True,exist_ok=False)
    state = dict(pid=os.getpid(),neural_forwards=0,optimizer_updates=0,automatic_training=False,automatic_rated=False)

    def save(phase,**fields):
        state.update(phase=phase,utc=datetime.now(timezone.utc).isoformat(),**fields)
        path=output/'state.json'
        path.with_suffix('.tmp').write_text(json.dumps(state,indent=2)+'\n')
        path.with_suffix('.tmp').replace(path)
        print(json.dumps(state),flush=True)

    try:
        save('select-forcing-branches')
        plan,exclusions,rows,counts,paths=prepare(source,output)
        root=Path(__file__).resolve().parents[1]
        runtime=root
        engines={r['name']:r for r in plan['opponents']}
        for name in ('weiss','stockfish'):
            assert artifact_info(engines[name]['path'])==engines[name]['artifact']
        paths += [Path(__file__).resolve(),root/'train/forcing.py',root/'chessio/records.py',root/'model/artifacts.py',
            *[Path(engines[n]['path']) for n in ('weiss','stockfish')],
            *[runtime/name for name in ('train/label_actions.py','train/uci_info.py','train/elite_labels.py','chessio/records.py','model/artifacts.py')]]
        frozen={str(p):artifact_info(p) for p in paths}
        requests=output/'requests.jsonl'
        save('independent-proposals',selected=len(rows),selection=counts,branch_kinds=dict(Counter(r['branch_kind'] for r in rows)))
        with chess.engine.SimpleEngine.popen_uci(engines['weiss']['path']) as weiss, \
             chess.engine.SimpleEngine.popen_uci(engines['stockfish']['path']) as sf, requests.open('x') as stream:
            assert weiss.id==engines['weiss']['engine_id'] and sf.id==engines['stockfish']['engine_id']
            weiss.configure(dict(Threads=1,Hash=64))
            sf.configure(dict(Threads=1,Hash=128,UCI_ShowWDL=True))
            for i,row in enumerate(rows,1):
                board=record_board(row)
                actor=weiss.play(board,chess.engine.Limit(depth=6),game=object(),info=chess.engine.INFO_ALL)
                screened=sf.play(board,chess.engine.Limit(nodes=512),game=object(),info=chess.engine.INFO_ALL)
                assert actor.move in board.legal_moves and screened.move in board.legal_moves
                row.update(actor_choice=actor.move.uci(),actor_choices=[actor.move.uci()],screened_teacher_move=screened.move.uci(),
                    actor_model=engines['weiss'],source_device='external-uci-proposal',
                    proposal_query=dict(depth_limit=6,reported_depth=actor.info.get('depth'),nodes=actor.info.get('nodes')),
                    screening_query=dict(nodes_limit=512,reported_depth=screened.info.get('depth'),nodes=screened.info.get('nodes')))
                print(json.dumps(row,separators=(',',':')),file=stream)
                if i%64==0:
                    stream.flush()
                    save('independent-proposals',prepared=i,total=len(rows))
        meta=dict(artifact=artifact_info(requests),inputs=frozen,exclusions=exclusions,label_runtime=str(runtime),
            requested=len(rows),seed=2026101913,selection=counts,min_both_recorded_rating=2800,
            source_game_cap=2,parent_history_plies=[16,160],branch_kinds=dict(Counter(r['branch_kind'] for r in rows)),
            proposal_only=True,actual_gpu_actor_review_pending=True,neural_forwards=0,optimizer_updates=0,
            automatic_training=False,automatic_rated=False)
        requests.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
        frozen.update({str(requests):artifact_info(requests),str(requests.with_suffix('.json')):artifact_info(requests.with_suffix('.json'))})
        assert all(artifact_info(p)==v for p,v in frozen.items())
        command=[sys.executable,'-u','-m','train.label_actions','--requests',str(requests),'--out',str(output/'labels.jsonl'),
            '--engine',engines['stockfish']['path'],'--unbounded','--hash-mib','256']
        with (output/'teacher.log').open('x') as log:
            child=subprocess.Popen(command,cwd=runtime,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONPATH':str(runtime),'OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'})
        save('independent-depth22-labels',child_pid=child.pid,teacher_command=command)
        assert child.wait()==0,'External teacher did not complete; partial results retained'
        result=label_review(output,frozen)
        assert result['requested']==4096
        (output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
        save('complete-forcing-label-review' if result['complete'] else 'incomplete-forcing-label-review',checked=result['checked'],counts=result['counts'])
    except BaseException as error:
        save('failed',error=repr(error))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.source.resolve(),args.out.resolve())
