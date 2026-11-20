import io
import json

import chess.pgn
import pytest

from scripts.build_human_corpus import Rated


@pytest.mark.parametrize('event,white,black,accepted', [
    ('Rated Blitz game', '2000', '2800', True),
    ('Rated Rapid game', '2400', '2400', True),
    ('Rated Bullet game', '2400', '2400', False),
    ('Rated Blitz game', '1999', '2400', False),
    ('Rated Blitz game', '2400', '2801', False),
    ('Rated Blitz game', '?', '2400', False),
])
def test_corpus_accepts_only_the_promised_rating_and_clock_pool(event, white, black, accepted):
    pgn = f'[Event "{event}"]\n[WhiteElo "{white}"]\n[BlackElo "{black}"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n'
    game = chess.pgn.read_game(io.StringIO(pgn), Visitor=Rated)
    assert bool(game.variations) == accepted


@pytest.mark.parametrize('kind', ['human', 'q'])
def test_builders_exclude_new_depth_puzzles_and_their_continuations(tmp_path, monkeypatch, kind):
    import zstandard
    from scripts import build_human_corpus, build_q_corpus

    monkeypatch.chdir(tmp_path)
    data = tmp_path / 'data'
    data.mkdir()
    for name in ('puzzles_holdout', 'puzzles_final3000', 'stockfish_holdout', 'eval_holdout', 'endgame_holdout'):
        (data / (name + '.jsonl')).write_text('')
    board = chess.Board()
    board.push_uci('e2e4')
    (data / 'puzzles_depth3000.jsonl').write_text(json.dumps({'fen': board.fen(), 'moves': ['e7e5']}) + '\n')
    blocked = [board.copy()]
    board.push_uci('e7e5')
    blocked.append(board.copy())
    output = data / 'training.jsonl'
    if kind == 'human':
        module = build_human_corpus
        plain = '[Event "Rated Blitz game"]\n[Site "local"]\n[WhiteElo "2400"]\n[BlackElo "2400"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 1-0\n'
        arguments = ['--month', '2024-01', '--positions', '2']
    else:
        module = build_q_corpus
        rows = [{'fen': b.fen(), 'evals': [{'depth': 14, 'pvs':
            [{'line': move.uci(), 'cp': 0} for move in b.legal_moves]}]} for b in blocked + [chess.Board()]]
        plain = ''.join(json.dumps(row) + '\n' for row in rows)
        arguments = ['--positions', '1']

    class Source(io.BytesIO):
        headers = {'ETag': 'frozen'}

    packed = zstandard.ZstdCompressor().compress(plain.encode())
    monkeypatch.setattr(module, 'RangeReader', lambda _: Source(packed))
    monkeypatch.setattr('sys.argv', ['build', *arguments, '--out', str(output)])
    module.main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    blocked_keys = {' '.join(b.fen().split()[:4]) for b in blocked}
    assert not any(' '.join(row['fen'].split()[:4]) in blocked_keys for row in rows)
    metadata = json.loads(output.with_suffix('.json').read_text())
    assert metadata['heldout_positions_skipped' if kind == 'human' else 'heldout_positions_excluded'] == 2
