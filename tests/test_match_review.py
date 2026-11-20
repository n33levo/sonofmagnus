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


@pytest.mark.parametrize('plies, protocol, valid', [(7, 'prospective_claim', True),
                                                  (7, 'present_position', False),
                                                  (8, 'present_position', True)])
def test_review_distinguishes_played_repetitions_from_legacy_prospective_claims(tmp_path, plies, protocol, valid):
    first, second = tmp_path / 'first.pt', tmp_path / 'second.pt'
    first.write_bytes(b'first model');second.write_bytes(b'second model')
    root = chess.STARTING_FEN
    openings = tmp_path / 'roots.fen'
    openings.write_text(root + '\n')
    games = []
    for white, black in [('first', 'second'), ('second', 'first')]:
        game = chess.pgn.Game()
        game.headers.update(White=white, Black=black, Result='1/2-1/2',
                            Termination='threefold_repetition', TimeControl='180+2')
        node = game
        for move in (['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2)[:plies]:
            node = node.add_variation(chess.Move.from_uci(move))
        games.append(str(game))
    result = dict(match_summary([.5, .5]), a=str(first), opponent_path=str(second),
                  model_a=artifact_info(first), opponent_artifact=artifact_info(second), runtime_source={},
                  clock_seconds=180, increment=2, openings_sha256=hashlib.sha256(root.encode()).hexdigest(),
                  moves_a=plies, moves_b=plies, neural_forward_calls_a=plies, neural_forward_calls_b=plies,
                  terminations={'threefold_repetition': 2})
    if protocol != 'prospective_claim':
        result['draw_claim_protocol'] = protocol
    path = tmp_path / 'match.json'
    path.write_text(json.dumps(result))
    path.with_suffix('.pgn').write_text('\n\n'.join(games) + '\n')
    if valid:
        assert review(path, openings)['legal_plies'] == 2 * plies
    else:
        with pytest.raises(AssertionError, match='Unfinished'):
            review(path, openings)


def test_review_flags_unverifiable_clock_forfeit(tmp_path):
    first, second = tmp_path / 'first.pt', tmp_path / 'second.pt'
    first.write_bytes(b'first model'); second.write_bytes(b'second model')
    root = '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'
    openings = tmp_path / 'roots.fen'
    openings.write_text(root + '\n')
    mate = chess.pgn.Game()
    mate.setup(chess.Board(root))
    mate.headers.update(White='first', Black='second', Result='1-0',
                        Termination='checkmate', TimeControl='180+2')
    mate.add_variation(chess.Move.from_uci('f7g7'))
    forfeit = chess.pgn.Game()
    forfeit.setup(chess.Board(root))
    forfeit.headers.update(White='second', Black='first', Result='0-1',
                           Termination='time_forfeit', TimeControl='180+2')
    result = dict(match_summary([1., 1.]), a=str(first), opponent_path=str(second),
                  model_a=artifact_info(first), opponent_artifact=artifact_info(second), runtime_source={},
                  clock_seconds=180, increment=2, openings_sha256=hashlib.sha256(root.encode()).hexdigest(),
                  moves_a=1, moves_b=1, neural_forward_calls_a=1, neural_forward_calls_b=1,
                  terminations={'checkmate': 1, 'time_forfeit': 1})
    path = tmp_path / 'match.json'
    path.write_text(json.dumps(result))
    path.with_suffix('.pgn').write_text(str(mate) + '\n\n' + str(forfeit) + '\n')
    with pytest.raises(AssertionError, match='Unfinished'):
        review(path, openings)
    audit = review(path, openings, allow_unverified_forfeits=True)
    assert audit['legal_plies'] == 1
    assert audit['neural_requests_by_side'] == [1, 1]
    assert audit['unverified_time_forfeits'] == 1
    assert not audit['outcomes_verified']
