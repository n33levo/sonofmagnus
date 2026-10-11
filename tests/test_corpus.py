import io

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
