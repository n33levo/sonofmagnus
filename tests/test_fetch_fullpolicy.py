import json
import pytest

import chess
import msgpack
import zstandard

from scripts.fetch_fullpolicy import extract, mate_order
from model.artifacts import artifact_info


def test_extract_requires_every_legal_move(tmp_path):
    board = chess.Board('8/1p1r1kp1/p4p1p/P7/1P5P/4B1P1/5P1K/8 b - - 0 44')
    moves = {move.uci(): {'win_prob': .5, 'mate': None} for move in board.legal_moves}
    first, last = list(moves)[0], list(moves)[-1]
    moves[first]['win_prob'], moves[last]['win_prob'] = 0., 1.
    bad = {'fen': board.fen(), 'moves': dict(list(moves.items())[1:])}
    good = {'fen': board.fen(), 'moves': moves}
    archive = tmp_path / 'sample.msgpack.zst'
    archive.write_bytes(zstandard.ZstdCompressor().compress(msgpack.packb(bad) + msgpack.packb(good)))
    excluded = tmp_path / 'holdout.jsonl'
    excluded.write_text('')
    out = tmp_path / 'training.jsonl'
    result = extract(archive, out, 1, [str(excluded)])
    row = json.loads(out.read_text())
    assert result['incomplete_filtered'] == 1
    assert set(row['q_values']) == set(moves)
    assert 0 < min(row['q_values'].values()) <= max(row['q_values'].values()) < 1
    assert result['source_archive'] == artifact_info(archive) and result['source'] is None
    assert result['history'].startswith('FEN only')


def test_equal_q_targets_keep_immediate_mate_and_its_distance(tmp_path):
    board = chess.Board('7k/5K2/6Q1/8/8/8/8/8 w - - 0 1')
    legal = list(board.legal_moves)
    mating = []
    for move in legal:
        board.push(move)
        if board.is_checkmate():
            mating.append(move.uci())
        board.pop()
    assert mating
    quiet = next(m.uci() for m in legal if m.uci() not in mating)
    # Put a slower annotated win first; Q alone cannot distinguish its mate distance.
    moves = {quiet: {'win_prob': 1., 'mate': 5}, **{m.uci(): {'win_prob': .5, 'mate': None} for m in legal if m.uci() != quiet}}
    for move in mating:
        moves[move] = {'win_prob': 1., 'mate': '#'}
    archive = tmp_path/'mates.zst'
    archive.write_bytes(zstandard.ZstdCompressor().compress(msgpack.packb({'fen': board.fen(), 'moves': moves})))
    out = tmp_path/'mates.jsonl'
    extract(archive, out, 1, [], min_pieces=3)
    row = json.loads(out.read_text())
    assert next(iter(row['move_probs'])) in mating
    assert row['mate_distances'][mating[0]] == '#'
    assert mate_order(2) > mate_order(5) and mate_order(-8) > mate_order(-2)
    with pytest.raises(ValueError, match='mate annotation'):
        mate_order(True)


def test_mirrored_holdout_is_excluded(tmp_path):
    board = chess.Board('8/1p1r1kp1/p4p1p/P7/1P5P/4B1P1/5P1K/8 b - - 0 44')
    moves = {m.uci(): {'win_prob': .5, 'mate': None} for m in board.legal_moves}
    archive = tmp_path/'positions.zst'
    archive.write_bytes(zstandard.ZstdCompressor().compress(msgpack.packb({'fen': board.fen(), 'moves': moves})))
    holdout = tmp_path/'holdout.jsonl'
    holdout.write_text(json.dumps({'fen': board.mirror().fen()})+'\n')
    out = tmp_path/'excluded.jsonl'
    with pytest.raises(ValueError, match='Incomplete extraction'):
        extract(archive, out, 1, [str(holdout)])
    assert not out.exists()
