"""Offline, independently scored action labels for frozen model proposals."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import chess.engine

from chessio.records import holdout_positions, record_board
from model.artifacts import artifact_info
from train.elite_labels import review_label
from train.uci_info import complete_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--parts', type=int, default=1)
    parser.add_argument('--reuse', type=Path)
    parser.add_argument('--unbounded', action='store_true', help='Complete depth22 without a wall-clock cap')
    parser.add_argument('--hash-mib', type=int, default=64)
    args = parser.parse_args()
    if not 0 <= args.part < args.parts:
        parser.error('Require 0 <= part < parts')
    if not 1 <= args.hash_mib <= 1024:
        parser.error('Hash must be between 1 and 1024 MiB')
    meta = json.loads(args.requests.with_suffix('.json').read_text())
    assert artifact_info(args.requests) == meta['artifact']
    blocked = holdout_positions(meta['exclusions'])
    paths = [args.requests, args.requests.with_suffix('.json'), args.engine, Path(__file__),
             Path(__file__).with_name('uci_info.py'), *map(Path, meta['exclusions'])]
    context = lambda r: (r['fen'], r.get('root_fen', r['fen']), tuple(r.get('history', [])))
    rows = list(map(json.loads, args.requests.open()))[args.part::args.parts]
    by_context = {context(r):r for r in rows}
    assert len(by_context) == len(rows)
    reused = {}
    if args.reuse:
        old = json.loads(args.reuse.with_suffix('.json').read_text())
        assert artifact_info(args.reuse) == old['artifact']
        resolve = lambda p: Path(p) if Path(p).is_absolute() else args.reuse.resolve().parent.parent/p
        assert all(artifact_info(resolve(p)) == v for p, v in old['inputs'].items())
        paths += [args.reuse, args.reuse.with_suffix('.json')]
        for row in map(json.loads, args.reuse.open()):
            review_label(row)
            key = context(row)
            if key in by_context:
                request = by_context[key]
                assert row['actor_choices'] == request['actor_choices']
                assert row['screened_teacher_move'] == request['screened_teacher_move']
                assert key not in reused
                reused[key] = row
    frozen = {str(p):artifact_info(p) for p in paths}
    counts, started = Counter(), time.monotonic()

    def observation(info, board):
        if info.get('depth', 0) < 22 or not info.get('pv') or info.get('lowerbound') or info.get('upperbound') or 'wdl' not in info:
            return None
        score, wdl = info['score'].pov(board.turn), info['wdl'].pov(board.turn)
        if score.is_mate():
            mate = score.mate()
            if not mate:
                return None
            cp, raw = (1 if mate > 0 else -1)*(100000-100*min(abs(mate), 999)), dict(mate=mate)
        else:
            cp, raw = score.score(), dict(cp=score.score())
        if type(cp) is not int or sum(wdl) != 1000:
            return None
        return dict(move=info['pv'][0].uci(), depth=info['depth'], cp=cp, raw=raw,
                    wdl=[wdl.wins, wdl.draws, wdl.losses], nodes=info.get('nodes'), seconds=info.get('time'))

    with chess.engine.SimpleEngine.popen_uci(str(args.engine)) as engine, args.out.with_suffix('.part').open('x') as stream:
        engine_id = dict(engine.id)
        assert engine_id.get('name') == 'Stockfish 19', 'This label protocol requires Stockfish19'
        options = {'Threads':1, 'Hash':args.hash_mib, 'UCI_ShowWDL':True}
        engine.configure(options)
        for index, request in enumerate(rows):
            board = record_board(request)
            legal = {m.uci() for m in board.legal_moves}
            actors = request['actor_choices']
            assert 1 <= len(actors) <= 2 and set(actors) <= legal and request['screened_teacher_move'] in legal
            assert not {' '.join(b.fen().split()[:4]) for b in (board, board.mirror())} & blocked
            result = reused.get(context(request))
            status = result['alignment_status'] if result else 'incomplete-unrestricted'
            if result is None:
                best = None
                for seconds in ((None,) if args.unbounded else (45, 90)):
                    info = complete_analysis(engine, board, chess.engine.Limit(depth=22, time=seconds), game=object())
                    best = observation(info, board)
                    if best:
                        break
                passes = [dict(kind='unrestricted', observed=best, reported_depth=info.get('depth', 0))]
                if best:
                    requested = list(dict.fromkeys([best['move'], *actors, request['screened_teacher_move']]))
                    scored = [best]
                    required = {best['move']}
                    if actors != [best['move']]:
                        required = set(requested)
                        for seconds in ((None,) if args.unbounded else (90, 180)):
                            infos = complete_analysis(engine, board, chess.engine.Limit(depth=22, time=seconds), multipv=len(requested),
                                root_moves=[board.parse_uci(m) for m in requested], game=object())
                            scored = [observation(v, board) for v in infos]
                            if all(v is not None for v in scored) and {v['move'] for v in scored} == required:
                                break
                        passes.append(dict(kind='restricted-actions', requested=requested, observed=scored,
                                           reported_depths=[v.get('depth', 0) for v in infos]))
                    status = 'incomplete-actions'
                    if all(v is not None for v in scored) and {v['move'] for v in scored} == required:
                        scores = {v['move']:v['cp'] for v in scored}
                        expert = max(scores, key=scores.get)
                        costs = {m:scores[expert]-v for m, v in scores.items()}
                        safe = [m for m, v in costs.items() if v <= 40]
                        wdl = next(v['wdl'] for v in scored if v['move'] == expert)
                        status = 'below-recovery' if scores[expert] < -300 else 'source-safe' if costs[actors[0]] <= 40 else 'confirmed-correction'
                        result = dict(request)
                        result.pop('reward_supervision', None)
                        result.update(move=expert, move_probs={m:1/len(safe) for m in safe},
                            value=(wdl[0]-wdl[2])/1000, value_source='Actual Stockfish root WDL: (wins-losses)/1000',
                            teacher_wdl=wdl, centipawns=scores, raw_scores={v['move']:v['raw'] for v in scored},
                            label_depth_min=min(v['depth'] for v in scored), complete_action_labels=set(scores) == legal,
                            actor_costs={m:costs[m] for m in actors}, alignment_status=status, teacher_observations=passes,
                            source='stockfish19-frozen-proposal-depth22' if status == 'confirmed-correction' else 'policy-prior-anchor',
                            weight=8 if status == 'confirmed-correction' else 1)
                        if status != 'below-recovery':
                            result['reward_supervision'] = dict(labelled_moves=list(costs), known_costs=list(costs.values()),
                                expert_depth_min=22, external_source='Fresh independent Stockfish19 labels on frozen proposals')
            if result is not None:
                assert review_label(result) == status
                print(json.dumps(result, separators=(',', ':')), file=stream)
                stream.flush()
            counts[status] += 1
            progress = dict(utc=datetime.now(timezone.utc).isoformat(), checked=index+1, total=len(rows),
                            counts=counts, seconds=time.monotonic()-started, reused=len(reused))
            args.out.with_suffix('.progress.json').write_text(json.dumps(progress, indent=2)+'\n')
            print(json.dumps(progress), flush=True)
    assert all(artifact_info(p) == v for p, v in frozen.items())
    args.out.with_suffix('.part').replace(args.out)
    args.out.with_suffix('.json').write_text(json.dumps(dict(
        inputs=frozen, artifact=artifact_info(args.out), counts=counts, exclusions=meta['exclusions'],
        requested=len(rows), reused=len(reused), engine_id=engine_id, uci_options=options,
        depth=22, wall_clock_cap=not args.unbounded,
        scope='Offline external action labels. Depth22 and actual WDL required for every requested move; incomplete roots excluded. No RL or online move source.'
    ), indent=2)+'\n')


if __name__ == '__main__':
    main()
