import hashlib
import json

import chess
import chess.pgn
import pytest

from eval.arena import match_summary
from model.artifacts import artifact_info
from scripts.review_match import review


def test_match_review_rejects_forged_ratings_and_neural_counts(tmp_path):
    first, second = tmp_path / 'first.pt', tmp_path / 'second.pt'
    first.write_bytes(b'first model')
    second.write_bytes(b'second model')
    root = '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'
    openings = tmp_path / 'roots.fen'
    openings.write_text(root + '\n')
    games = []
    for white, black in [('first', 'second'), ('second', 'first')]:
        game = chess.pgn.Game()
        game.setup(chess.Board(root))
        game.headers.update(White=white, Black=black, Result='1-0', Termination='checkmate', TimeControl='10+0.1')
        game.add_variation(chess.Move.from_uci('f7g7'))
        games.append(str(game))
    result = dict(match_summary([1., 0.]), a=str(first), opponent_path=str(second),
                  model_a=artifact_info(first), opponent_artifact=artifact_info(second), runtime_source={},
                  clock_seconds=10, increment=.1, openings_sha256=hashlib.sha256(root.encode()).hexdigest(),
                  moves_a=1, moves_b=1, neural_forward_calls_a=1, neural_forward_calls_b=1,
                  terminations={'checkmate': 2})
    path = tmp_path / 'match.json'
    path.write_text(json.dumps(result))
    path.with_suffix('.pgn').write_text('\n\n'.join(games) + '\n')
    assert review(path, openings)['legal_plies'] == 2
    result['elo_difference'] += 100
    path.write_text(json.dumps(result))
    with pytest.raises(AssertionError, match='Rating calculation differs'):
        review(path, openings)
    result['elo_difference'] -= 100
    result['neural_forward_calls_b'] += 1
    path.write_text(json.dumps(result))
    with pytest.raises(AssertionError):
        review(path, openings)
