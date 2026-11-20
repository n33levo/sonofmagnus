"""Score recorded moves on their full Lichess histories with a fixed external budget."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path

import chess
import chess.engine
import numpy as np

from chessio.records import record_board
from model.artifacts import artifact_info


def loss(best, played):
    # Winning mates are equivalent outcomes; their distances are not pawn losses.
    if (best.get('mate') or 0) > 0 and (played.get('mate') or 0) > 0:
        cp = 0
    elif best.get('mate') is not None or played.get('mate') is not None:
        cp = 600 if best['points'] > played['points'] else 0
    else:
        cp = max(0, min(600, best['cp'] - played['cp']))
    return dict(point_loss=max(0., best['points'] - played['points']), cp_loss=cp,
                forced_exceeds_best=played['points'] > best['points'])


def observe(engine, board, nodes, move, key):
    kwargs = dict(game=key, info=chess.engine.INFO_ALL)
    if move is not None:
        kwargs['root_moves'] = [chess.Move.from_uci(move)]
    latest, final_nodes = None, 0
    with engine.analysis(board, chess.engine.Limit(nodes=nodes), **kwargs) as stream:
        for info in stream:
            final_nodes = max(final_nodes, info.get('nodes', 0))
            if not all(name in info for name in ('score', 'wdl', 'pv')) or not info['pv']:
                continue
            if info.get('lowerbound') or info.get('upperbound'):
                continue
            score, wdl = info['score'].pov(board.turn), info['wdl'].pov(board.turn)
            pv = [item.uci() for item in info['pv']]
            if move is not None and pv[0] != move:
                raise ValueError('Forced analysis returned another root move')
            latest = dict(move=pv[0], cp=score.score(), mate=score.mate(),
                wdl=list(wdl), points=(wdl.wins + .5 * wdl.draws) / 1000,
                depth=info.get('depth'), iteration_nodes=info.get('nodes'), pv=pv)
    if latest is None:
        raise ValueError('No complete, unbounded score/WDL/PV observation')
    latest['search_nodes'] = final_nodes
    return latest


def score_game(task):
    game_id, roots, choices, engine_path, nodes, cache_folder = task
    path = Path(cache_folder) / (game_id + '.jsonl')
    results = []
    with path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cache = {}
        if path.exists():
            for line in path.read_text().splitlines():
                item = json.loads(line)
                key = (item['ply'], item['requested_move'])
                if key in cache and cache[key] != item:
                    raise ValueError('Conflicting cached observations')
                cache[key] = item
        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine, path.open('a') as saved:
            engine.configure({'Threads': 1, 'Hash': 128, 'UCI_ShowWDL': True})
            for root in roots:
                board = record_board(root)

                def query(move):
                    key = (root['ply'], move)
                    if key not in cache:
                        observed = observe(engine, board, nodes, move, (game_id, *key))
                        cache[key] = dict(ply=root['ply'], requested_move=move, fen=root['fen'], **observed)
                        saved.write(json.dumps(cache[key], separators=(',', ':')) + '\n')
                        saved.flush()
                    item = cache[key]
                    if item['fen'] != root['fen']:
                        raise ValueError('Cached observation belongs to another root')
                    return item

                best = query(None)
                actions = dict(played=root['played_move'])
                if choices is not None:
                    actions['candidate'] = choices[(game_id, root['ply'])]
                scores = {}
                for name, move in actions.items():
                    if chess.Move.from_uci(move) not in board.legal_moves:
                        raise ValueError('Illegal scored move')
                    observed = best if move == best['move'] else query(move)
                    scores[name] = dict(move=move, observed=observed, **loss(best, observed))
                results.append(dict(game_id=game_id, ply=root['ply'], own_move=root['own_move'],
                    pieces=len(board.piece_map()), best=best, scores=scores))
    return results


def run(args):
    data = artifact_info(args.positions)
    roots = list(map(json.loads, args.positions.open()))
    if len({(row['game_id'], row['ply']) for row in roots}) != len(roots):
        raise ValueError('Duplicate own-move roots')
    grouped = defaultdict(list)
    for row in roots:
        grouped[row['game_id']].append(row)
    choices = None
    if args.choices:
        metadata = json.loads(args.choices.with_suffix('.json').read_text())
        if metadata['artifact'] != artifact_info(args.choices) or metadata['positions'] != data:
            raise ValueError('Candidate inference artifacts changed')
        if metadata['roots'] != metadata['neural_forwards'] or metadata['roots'] != len(roots):
            raise ValueError('Candidate did not use one forward per root')
        entries = list(map(json.loads, args.choices.open()))
        choices = {(row['game_id'], row['ply']): row['move'] for row in entries}
        if len(choices) != len(entries) or set(choices) != {(row['game_id'], row['ply']) for row in roots}:
            raise ValueError('Candidate choices must cover exactly the same roots')
        for row, entry in zip(roots, entries):
            if (row['game_id'], row['ply'], row['fen']) != (entry['game_id'], entry['ply'], entry['fen']):
                raise ValueError('Candidate inference context mismatch')
    args.cache.mkdir(parents=True, exist_ok=True)
    protocol = dict(positions=data, engine=artifact_info(args.engine), nodes=args.nodes, threads=1, hash_mib=128,
        fresh_hash_each_query=True, producer=artifact_info(__file__),
        observations='Last complete exact score, actual WDL and PV within a fixed node budget. No training labels or serving move sources.')
    protocol_path = args.cache / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Search cache protocol changed; use another cache directory')
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    args.out.mkdir(parents=True, exist_ok=False)
    all_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_game, (game, rows, choices, str(args.engine), args.nodes, str(args.cache)))
                   for game, rows in grouped.items()]
        for completed, future in enumerate(as_completed(futures), 1):
            results = future.result()
            all_results.extend(results)
            (args.out / (results[0]['game_id'] + '.json')).write_text(json.dumps(results) + '\n')
            print(json.dumps(dict(completed_games=completed, total_games=len(grouped),
                completed_roots=len(all_results), total_roots=len(roots))), flush=True)
    assert len(all_results) == len(roots) and artifact_info(args.positions) == data
    summary = {}
    for name in ('played', 'candidate') if choices is not None else ('played',):
        values = [row['scores'][name] for row in all_results]
        summary[name] = dict(roots=len(values), mean_point_loss=float(np.mean([v['point_loss'] for v in values])),
            mean_cp_loss_capped600=float(np.mean([v['cp_loss'] for v in values])),
            errors100=sum(v['cp_loss'] >= 100 for v in values), errors300=sum(v['cp_loss'] >= 300 for v in values),
            forced_exceeds_unrestricted=sum(v['forced_exceeds_best'] for v in values))
    report = dict(games=len(grouped), roots=len(roots), summary=summary, protocol=protocol,
        positions=str(args.positions), choices=artifact_info(args.choices) if args.choices else None,
        automatic_rated=False, training_allowed=False, target_achieved=False,
        scope='All scored roots are retrospective actual game histories. Fixed-budget teacher scores can be noisy; they do not prove alternate game outcomes or an official Elo.')
    (args.out / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--positions', type=Path, default=Path('data/lichess-history/positions.jsonl'))
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--nodes', type=int, default=100000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--choices', type=Path)
    options = parser.parse_args()
    if options.nodes < 1 or not 1 <= options.workers <= 2:
        parser.error('Positive node budget and one or two workers required')
    run(options)
