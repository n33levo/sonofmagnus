import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import chess

from eval.arena import UciRunner, play_match
from eval.rating import summarize_pool
from model.artifacts import artifact_info
from play.runner import load_runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--pool', default='data/arena_pool.json')
    parser.add_argument('--openings', default='data/arena_openings.fen')
    parser.add_argument('--games', type=int, default=200, help='Games per opponent, with both colours')
    parser.add_argument('--time-control', default='30+0.3')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    pool = json.loads(Path(args.pool).read_text())
    openings = [line.strip() for line in Path(args.openings).read_text().splitlines() if line.strip()]
    if args.games <= 0 or args.games % 2 or args.games // 2 > len(openings):
        parser.error('Use an even game count with no repeated opening pairs')
    keys = [' '.join(chess.Board(fen).fen().split()[:4]) for fen in openings[:args.games // 2]]
    if len(set(keys)) != len(keys):
        parser.error('Opening pairs must have distinct positions')
    if any(not Path(engine['path']).is_file() for engine in pool['engines']):
        parser.error('Build the opponents first: python -m scripts.build_pool')
    clock, increment = map(float, args.time_control.split('+'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    runner = load_runner(args.ckpt)
    for _ in range(10):
        runner.policy(chess.Board())
    frozen = artifact_info(args.ckpt)
    manifest = {'checkpoint': args.ckpt, 'model': frozen, 'pool': pool, 'platform': platform.platform(),
                'machine': platform.machine(), 'time_control': args.time_control,
                'openings_sha256': hashlib.sha256(Path(args.openings).read_bytes()).hexdigest(),
                'tablebases': False, 'opponents': [], 'games_per_opponent': args.games,
                'binaries': {engine['name']: artifact_info(engine['path']) for engine in pool['engines']},
                'runtime_source': {path: artifact_info(path)['sha256'] for path in
                    ('play/mlx.py', 'play/runner.py', 'chessio/lc0.py', 'eval/arena.py', 'eval/rating.py', 'eval/pool.py',
                     'chessio/encode.py', 'chessio/mask.py', 'chessio/policy_map.py',
                     'model/__init__.py', 'model/resnet.py', 'model/heads.py')}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    scores = []
    for specification in pool['engines']:
        engine = UciRunner(specification['path'])
        pgn_path = out / (specification['name'] + '.pgn')
        result_path = out / (specification['name'] + '.json')
        try:
            result = play_match(runner, engine, args.games, openings=openings, max_plies=1000,
                                pgn_path=pgn_path, clock_seconds=clock, increment=increment,
                                names=(Path(args.ckpt).stem, specification['name']))
            binary = artifact_info(specification['path'])
            if binary != manifest['binaries'][specification['name']]:
                raise ValueError('Opponent binary changed during tournament')
            result.update(opponent=specification, binary=binary,
                          engine_id=engine.engine.id, uci_options=engine.configured_options)
            result_path.write_text(json.dumps(result, indent=2) + '\n')
            if result['terminations'].get('ply_limit'):
                raise ValueError('Unfinished games reached the ply cap; no rating will be published')
            if result.get('neural_forward_calls_a') != result['moves_a']:
                raise ValueError('Expected exactly one neural forward pass per move')
            scores.append(result['scores_a'])
            manifest['opponents'].append(specification['name'])
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        finally:
            engine.close()
    if artifact_info(args.ckpt) != frozen:
        raise ValueError('Checkpoint changed during tournament')
    summary = summarize_pool([list(zip(row[::2], row[1::2])) for row in scores],
                             [engine['reference_elo'] for engine in pool['engines']])
    summary['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (out / 'rating.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
