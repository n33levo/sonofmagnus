import json

import chess
import chess.pgn
import pytest

from eval.rating import summarize_pool
from model.artifacts import artifact_info
from scripts.review_pool import review


def test_pool_review_replays_games_and_rejects_changed_scores_and_hashes(tmp_path):
    checkpoint, opponent = tmp_path / 'student.pt', tmp_path / 'opponent'
    checkpoint.write_bytes(b'frozen model')
    opponent.write_bytes(b'frozen opponent')
    specification = {'name': 'opponent', 'path': str(opponent), 'reference_elo': 2500}
    manifest = {'checkpoint': str(checkpoint), 'model': artifact_info(checkpoint),
                'pool': {'engines': [specification]}, 'binaries': {'opponent': artifact_info(opponent)},
                'runtime_source': {}, 'games_per_opponent': 2, 'time_control': '10+0.1'}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    games = []
    for white, black in [('student', 'opponent'), ('opponent', 'student')]:
        board = chess.Board('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1')
        game = chess.pgn.Game()
        game.setup(board)
        game.headers.update(White=white, Black=black, Result='1-0', Termination='checkmate', TimeControl='10+0.1')
        node = game.add_variation(chess.Move.from_uci('f7g7'))
        node.comment = '10.000 ms'
        games.append(str(game))
    (tmp_path / 'opponent.pgn').write_text('\n\n'.join(games) + '\n')
    result = {'games': 2, 'scores_a': [1., 0.], 'terminations': {'checkmate': 2},
              'moves_a': 1, 'neural_forward_calls_a': 1}
    result_path = tmp_path / 'opponent.json'
    result_path.write_text(json.dumps(result))
    rating_path = tmp_path / 'rating.json'
    rating = summarize_pool([[[1., 0.]]], [2500])
    rating_path.write_text(json.dumps(rating))
    checked = review(tmp_path)
    assert checked['games'] == checked['legal_plies'] == 2
    assert checked['engine_moves'] == 1 and checked['latency_ms']['p95'] == 10
    result['scores_a'][0] = .5
    result_path.write_text(json.dumps(result))
    with pytest.raises(AssertionError):
        review(tmp_path)
    result['scores_a'][0] = 1.
    result_path.write_text(json.dumps(result))
    rating['local_elo'] += 100
    rating_path.write_text(json.dumps(rating))
    with pytest.raises(AssertionError, match='Rating calculation differs'):
        review(tmp_path)
    checkpoint.write_bytes(b'changed model')
    with pytest.raises(AssertionError, match='Checkpoint changed'):
        review(tmp_path)
