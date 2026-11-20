import chess
import pytest

from chessio.records import record_board
from scripts.prepare_mates import mating_rows,puzzle_split


def example():
    return dict(PuzzleId='00sHx',FEN='q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17',
                Moves='e8d7 a2e6 d7d8 f7f8',Themes='mate mateIn2 middlegame short',Rating='1760')


def test_mating_rows_keep_only_solver_decisions_and_past_history():
    rows,keys=mating_rows(example())
    assert len(rows)==2 and [r['mate_moves_remaining'] for r in rows]==[2,1]
    assert [len(r['history']) for r in rows]==[1,3]
    assert [r['move'] for r in rows]==['a2e6','f7f8']
    for row in rows:
        board=record_board(row)
        assert board.turn==chess.WHITE and row['game_id']=='puzzle-00sHx'
        assert ' '.join(board.mirror().fen().split()[:4]) in keys
    last=record_board(rows[-1]);last.push_uci(rows[-1]['move'])
    assert last.is_checkmate() and ' '.join(last.fen().split()[:4]) in keys


@pytest.mark.parametrize('update', [dict(Moves='e8d7 a2e6 d7d8 f7f6'),
    dict(Moves='e8d7 a2e6 d7d8 a1a8'),dict(Themes='mate mateIn3'),
    dict(Moves='e8d7 a2e6 d7d8'),dict(Themes='mate mateIn1 mateIn2')])
def test_invalid_or_incomplete_mating_lines_are_rejected(update):
    with pytest.raises(ValueError): mating_rows(dict(example(),**update))


def test_split_is_reproducible_and_covers_both_roles():
    choices={puzzle_split(str(i),7) for i in range(200)}
    assert choices=={'fit','validation'}
    for i in range(200): assert puzzle_split(str(i),7)==puzzle_split(str(i),7)


def test_nullable_game_id_does_not_publish_an_incomplete_scan(tmp_path, monkeypatch):
    import json
    import zstandard
    from scripts.prepare_mates import main

    archive=tmp_path/'puzzles.csv.zst'
    archive.write_bytes(zstandard.ZstdCompressor().compress(b'PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes\n'))
    excluded=tmp_path/'excluded.jsonl'
    excluded.write_text(json.dumps(dict(fen=chess.STARTING_FEN,game_id=None))+'\n')
    prefix=tmp_path/'mates'
    monkeypatch.setattr('sys.argv',['prepare_mates','--archive',str(archive),'--exclude-data',str(excluded),
        '--out',str(prefix),'--fit-per-band','1','1','1','1','--validation-per-band','1','1','1','1'])
    with pytest.raises(RuntimeError,match='Archive exhausted'):
        main()
    assert not prefix.with_suffix('.json').exists()
    assert not prefix.with_suffix('.fit.jsonl').exists()
    assert not prefix.with_suffix('.validation.jsonl').exists()
