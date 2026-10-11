import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import chess.pgn
import numpy as np

from model.artifacts import artifact_info
from eval.rating import summarize_pool


def review(directory, source_commit=None):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert artifact_info(manifest['checkpoint']) == manifest['model'], 'Checkpoint changed'
    for name, expected in manifest['runtime_source'].items():
        actual = (hashlib.sha256(subprocess.check_output(['git', 'show', f'{source_commit}:{name}'])).hexdigest()
                  if source_commit else artifact_info(name)['sha256'])
        assert actual == expected, f'Runtime source changed: {name}'
    total = Counter()
    matches, latencies, pair_scores = {}, [], []
    for engine in manifest['pool']['engines']:
        name = engine['name']
        assert artifact_info(engine['path']) == manifest['binaries'][name], 'Opponent changed'
        result = json.loads((directory / f'{name}.json').read_text())
        counts, reasons, scores, openings = Counter(), Counter(), [], []
        with (directory / f'{name}.pgn').open() as stream:
            while (game := chess.pgn.read_game(stream)) is not None:
                assert not game.errors
                index = counts['games']
                a_white = index % 2 == 0
                assert game.headers['White' if a_white else 'Black'] == Path(manifest['checkpoint']).stem
                assert game.headers['TimeControl'] == manifest['time_control']
                board = game.board()
                openings.append(board.fen())
                for node in game.mainline():
                    assert node.move in board.legal_moves
                    if board.turn == a_white:
                        counts['engine_moves'] += 1
                        elapsed, unit = node.comment.split()
                        assert unit == 'ms' and float(elapsed) >= 0
                        latencies.append(float(elapsed))
                    board.push(node.move)
                    counts['legal_plies'] += 1
                outcome = board.outcome(claim_draw=True)
                assert outcome is not None, 'Unfinished game or forfeit'
                assert game.headers['Result'] == outcome.result()
                assert game.headers['Termination'] == outcome.termination.name.lower()
                scores.append(.5 if outcome.winner is None else float(outcome.winner == a_white))
                reasons[game.headers['Termination']] += 1
                counts['games'] += 1
        assert counts['games'] == manifest['games_per_opponent'] == result['games']
        assert openings[::2] == openings[1::2], 'Colour pair used different openings'
        assert len(set(openings[::2])) == counts['games'] // 2
        assert scores == result['scores_a']
        assert dict(reasons) == result['terminations']
        assert counts['engine_moves'] == result['moves_a'] == result['neural_forward_calls_a']
        pair_scores.append(list(zip(scores[::2], scores[1::2])))
        total.update(counts)
        matches[name] = dict(counts, pgn=artifact_info(directory / f'{name}.pgn'))
    rating = json.loads((directory / 'rating.json').read_text())
    assert rating['games'] == total['games']
    calculated = summarize_pool(pair_scores, [engine['reference_elo'] for engine in manifest['pool']['engines']])
    for name, expected in calculated.items():
        assert rating[name] == expected, f'Rating calculation differs: {name}'
    return dict(total, matches=matches, model=manifest['model'],
                latency_ms={f'p{q}': float(np.percentile(latencies, q)) for q in (50, 95, 99)},
                outcomes_verified=True, source_hashes_verified=True, forfeits=0, ply_caps=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--source-commit')
    args = parser.parse_args()
    result = review(args.directory, args.source_commit)
    Path(args.directory, 'independent_review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
