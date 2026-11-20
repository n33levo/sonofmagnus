import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import chess.pgn

from eval.arena import match_summary
from model.artifacts import artifact_info


def review(path, openings_path, source_commit=None, allow_unverified_forfeits=False):
    path = Path(path)
    result = json.loads(path.read_text())
    assert result['games'] > 0 and result['games'] % 2 == 0
    protocol = result.get('draw_claim_protocol', 'prospective_claim')
    assert protocol in ('present_position', 'prospective_claim')
    frozen = artifact_info(path), artifact_info(path.with_suffix('.pgn'))
    assert artifact_info(result['a']) == result['model_a']
    assert artifact_info(result['opponent_path']) == result['opponent_artifact']
    for name, expected in result['runtime_source'].items():
        actual = (hashlib.sha256(subprocess.check_output(['git', 'show', f'{source_commit}:{name}'])).hexdigest()
                  if source_commit else artifact_info(name)['sha256'])
        assert actual == expected, f'Runtime source changed: {name}'
    openings = [line.strip() for line in Path(openings_path).read_text().splitlines() if line.strip()]
    assert hashlib.sha256('\n'.join(openings).encode()).hexdigest() == result['openings_sha256']
    roots = [chess.Board(fen).fen() for fen in openings[:result['games'] // 2]]
    assert len(roots) == result['games'] // 2
    assert len({' '.join(fen.split()[:4]) for fen in roots}) == len(roots)
    games, moves, scores, reasons, unverified_forfeits = 0, [0, 0], [], Counter(), [0, 0]
    with path.with_suffix('.pgn').open() as stream:
        while (game := chess.pgn.read_game(stream)) is not None:
            assert not game.errors and games < result['games']
            a_white = games % 2 == 0
            assert game.headers['White' if a_white else 'Black'] == Path(result['a']).stem
            board = game.board()
            assert board.fen() == roots[games // 2]
            if result['clock_seconds'] is not None:
                assert game.headers['TimeControl'] == f"{result['clock_seconds']:g}+{result['increment']:g}"
            for move in game.mainline_moves():
                assert move in board.legal_moves
                moves[int(board.turn != a_white)] += 1
                board.push(move)
            outcome = board.outcome(claim_draw=protocol == 'prospective_claim')
            if outcome is None and protocol == 'present_position':
                if board.is_repetition(3):
                    outcome = chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
                elif board.is_fifty_moves():
                    outcome = chess.Outcome(chess.Termination.FIFTY_MOVES, None)
            if outcome is None:
                assert (allow_unverified_forfeits and result['clock_seconds'] is not None and
                        game.headers['Termination'] == 'time_forfeit' and
                        game.headers['Result'] in ('1-0', '0-1')), 'Unfinished game or forfeit'
                assert (game.headers['Result'] == '1-0') != board.turn
                scores.append(float((game.headers['Result'] == '1-0') == a_white))
                unverified_forfeits[int(board.turn != a_white)] += 1
            else:
                assert game.headers['Result'] == outcome.result()
                assert game.headers['Termination'] == outcome.termination.name.lower()
                scores.append(.5 if outcome.winner is None else float(outcome.winner == a_white))
            reasons[game.headers['Termination']] += 1
            games += 1
    assert games == result['games'] and scores == result['scores_a']
    assert dict(reasons) == result['terminations']
    requests = [moves[i] + unverified_forfeits[i] for i in (0, 1)]
    assert requests == [result['moves_a'], result['moves_b']]
    assert requests[0] == result['neural_forward_calls_a']
    if 'neural_forward_calls_b' in result:
        assert requests[1] == result['neural_forward_calls_b']
    for name, expected in match_summary(scores).items():
        assert result[name] == expected, f'Rating calculation differs: {name}'
    assert frozen == (artifact_info(path), artifact_info(path.with_suffix('.pgn')))
    return dict(games=games, legal_plies=sum(moves), moves_by_side=moves,
                neural_requests_by_side=requests, outcomes_verified=sum(unverified_forfeits) == 0,
                unverified_time_forfeits=sum(unverified_forfeits),
                distinct_colour_pairs=len(roots), artifacts=frozen, source_hashes_verified=True,
                rating_calculation_verified=True, reviewer_sha256=artifact_info(__file__)['sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('match')
    parser.add_argument('--openings', required=True)
    parser.add_argument('--source-commit')
    parser.add_argument('--allow-unverified-forfeits', action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = review(args.match, args.openings, args.source_commit, args.allow_unverified_forfeits)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
