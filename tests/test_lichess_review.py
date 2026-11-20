import copy
import io
import json

import chess
import pytest

from scripts.review_lichess import has_rating_update, rated_game_count, review


@pytest.mark.parametrize('failure', [None, 'calls', 'missing', 'outcome', 'duplicate'])
def test_server_games_replay_against_every_neural_decision(failure):
    board, records = chess.Board(), []
    for text in ['f2f3', 'e7e5', 'g2g4', 'd8h4']:
        move = board.parse_uci(text)
        if not board.turn:
            records.append({'root_fen': chess.STARTING_FEN, 'fen': board.fen(),
                'history': [m.uci() for m in board.move_stack], 'move': text,
                'neural_forward_calls': 1, 'latency_ms': 10.})
        board.push(move)
    game = {'id': 'testgame', 'status': 'mate', 'rated': True, 'variant': 'standard', 'speed': 'blitz',
            'winner': 'black', 'players': {'white': {'user': {'id': 'opponent'}}, 'black': {'user': {'id': 'n33levo'}}},
            'clock': {'initial': 180, 'increment': 2},
            'moves': 'f3 e5 g4 Qh4#',
            'pgn': '[Result "0-1"]\n\n1. f3 e5 2. g4 Qh4# 0-1\n'}
    games = [game]
    if failure == 'calls':
        records[0]['neural_forward_calls'] = 2
    elif failure == 'missing':
        records.pop()
    elif failure == 'outcome':
        game['winner'] = 'white'
    elif failure == 'duplicate':
        games.append(copy.deepcopy(game))
    if failure:
        with pytest.raises(ValueError):
            review(games, records, 'n33levo')
    else:
        result = review(games, records, 'n33levo')
        assert result['games'] == result['wins'] == 1
        assert result['legal_plies'] == 4
        assert result['neural_moves_verified'] == 2
        assert result['unmatched_decisions'] == 0


def draw_game():
    game = chess.pgn.Game()
    game.headers['Result'] = '1/2-1/2'
    node, board, records = game, game.board(), []
    for ply, text in enumerate(['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2):
        move = board.parse_uci(text)
        if board.turn:
            records.append({'root_fen': chess.STARTING_FEN, 'fen': board.fen(),
                'history': [m.uci() for m in board.move_stack], 'move': text,
                'neural_forward_calls': 1, 'latency_ms': 10., 'utc_seconds': 100 + ply + .01})
        node = node.add_variation(move)
        board.push(move)
    item = {'id': 'drawgame', 'status': 'draw', 'rated': True, 'variant': 'standard', 'speed': 'blitz',
        'players': {'white': {'user': {'id': 'n33levo'}}, 'black': {'user': {'id': 'opponent'}}},
        'clock': {'initial': 180, 'increment': 2}, 'createdAt': 100000, 'lastMoveAt': 108000,
        'moves': 'Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8', 'pgn': str(game)}
    return item, records, board


@pytest.mark.parametrize('known_root', [True, False])
def test_only_exact_finished_root_requests_are_classified(known_root):
    game, records, board = draw_game()
    if not known_root:
        board.push_uci('e2e4')
    extra = {'root_fen': chess.STARTING_FEN, 'fen': board.fen(),
        'history': [move.uci() for move in board.move_stack], 'move': next(iter(board.legal_moves)).uci(),
        'neural_forward_calls': 1, 'latency_ms': 10., 'utc_seconds': 108.1}
    result = review([game], [*records, extra], 'n33levo')
    assert result['games'] == result['draws'] == 1
    assert result['neural_moves_verified'] == 4
    assert len(result['cancelled_decisions']) == int(known_root)
    assert result['unmatched_decisions'] == int(not known_root)


def test_game_windows_separate_an_aborted_first_move_from_the_same_later_move():
    game, records, _ = draw_game()
    partial = chess.pgn.Game()
    partial.add_variation(chess.Move.from_uci('g1f3'))
    aborted = copy.deepcopy(game)
    aborted.update(id='aborted', status='aborted', rated=False, createdAt=90000,
                   lastMoveAt=90000, moves='Nf3', pgn=str(partial))
    cancelled = {**records[0], 'utc_seconds': 90.1}
    result = review([aborted, game], [cancelled, *records], 'n33levo')
    assert result['games'] == result['draws'] == 1
    assert result['aborted_game_ids'] == ['aborted']
    assert result['neural_moves_verified'] == 4
    assert len(result['cancelled_decisions']) == 1
    assert result['unmatched_decisions'] == 0


def test_live_sample_counts_finished_exports_instead_of_profile_games(monkeypatch):
    rows = [{'rated': rated, 'variant': 'standard', 'speed': 'blitz', 'status': status,
             'players': {'white': {'user': {'id': 'n33levo'}, 'ratingDiff': diff}}}
            for rated, status, diff in [(True, 'mate', -12), (True, 'draw', 0), (True, 'started', 1),
                                      (False, 'mate', 1), (True, 'aborted', 1), (True, 'mate', None)]]
    def response(request, timeout, context):
        assert context.check_hostname
        assert 'since=12345' in request.full_url and 'ongoing=false' in request.full_url
        assert request.headers['Authorization'] == 'Bearer test-token'
        return io.BytesIO('\n'.join(json.dumps(row) for row in rows).encode())
    monkeypatch.setattr('scripts.review_lichess.urllib.request.urlopen', response)
    assert rated_game_count('n33levo', 12345, 'test-token') == 2


@pytest.mark.parametrize('diff, expected', [(None, False), (0, True), (-12, True), (False, False)])
def test_rating_update_requires_the_accounts_explicit_numeric_difference(diff, expected):
    game = {'players': {'white': {'user': {'id': 'n33levo'}, 'ratingDiff': diff},
                        'black': {'user': {'id': 'opponent'}, 'ratingDiff': 10}}}
    assert has_rating_update(game, 'N33LEVO') is expected


def test_training_replay_contains_only_accepted_moves_in_completed_games():
    game, records, board = draw_game()
    extra = {**records[0], 'fen': board.fen(), 'history': [m.uci() for m in board.move_stack],
             'utc_seconds': 108.1}
    played = []
    audit = review([game], [*records, extra], 'n33levo', played_records=played)
    assert audit['unmatched_decisions'] == 0 and len(audit['cancelled_decisions']) == 1
    assert played == [('drawgame', row) for row in records]


def test_training_replay_excludes_aborted_moves_even_when_the_opening_is_identical():
    game, records, _ = draw_game()
    partial = chess.pgn.Game()
    partial.add_variation(chess.Move.from_uci('g1f3'))
    aborted = {**game, 'id': 'aborted', 'status': 'aborted', 'rated': False,
               'createdAt': 90000, 'lastMoveAt': 90000, 'moves': 'Nf3', 'pgn': str(partial)}
    played = []
    audit = review([aborted, game], [{**records[0], 'utc_seconds': 90.1}, *records],
                   'n33levo', played_records=played)
    assert audit['unmatched_decisions'] == 0
    assert played == [('drawgame', row) for row in records]


@pytest.mark.parametrize('changed', [None, 'source', 'model'])
def test_live_replay_binds_serving_artifacts_and_excludes_holdout_positions(tmp_path, changed):
    from model.artifacts import artifact_info
    from scripts.live_replay import export
    game, records, board = draw_game()
    extra = {**records[0], 'fen': board.fen(), 'history': [m.uci() for m in board.move_stack],
             'utc_seconds': 108.1}
    run = tmp_path / 'run'
    (run / 'runtime').mkdir(parents=True)
    (run / 'games').mkdir()
    source, model = run / 'runtime/source.py', run / 'model.mlx'
    source.write_text('frozen serving code')
    model.write_bytes(b'frozen test weights')
    manifest = dict(runtime_directory='runtime', runtime_source={'source.py': artifact_info(source)['sha256']},
                    source_checkpoint='ckpts/model.mlx', model=artifact_info(model), account='n33levo')
    (run / 'manifest.json').write_text(json.dumps(manifest))
    (run / 'server_games.jsonl').write_text(json.dumps(game) + '\n')
    (run / 'moves.jsonl').write_text('\n'.join(json.dumps(row) for row in [*records, extra]) + '\n')
    heldout = tmp_path / 'heldout.jsonl'
    heldout.write_text(json.dumps({'fen': records[1]['fen']}) + '\n')
    output = tmp_path / 'replay.jsonl'
    if changed:
        (source if changed == 'source' else model).write_bytes(b'changed')
        with pytest.raises(ValueError, match='Frozen serving'):
            export(run, output, [str(heldout)])
        assert not output.exists()
    else:
        result = export(run, output, [str(heldout)])
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert result['accepted_decisions'] == 4 and result['cancelled_decisions'] == 1
        assert result['positions'] == 2 and result['excluded_decisions'] == 2
        assert [row['history'] for row in rows] == [records[0]['history'], records[2]['history']]
        assert {row['game_id'] for row in rows} == {'drawgame'}
        assert not any('move_probs' in row or 'q_values' in row for row in rows)
