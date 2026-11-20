import json
import random

import chess
import pytest

from model.artifacts import artifact_info
from scripts.sample_arena import sample


def test_new_arena_sample_excludes_earlier_files_without_a_manual_list(tmp_path):
    source = tmp_path / 'games.jsonl'
    rng, games = random.Random(42), []
    for _ in range(30):
        board, moves = chess.Board(), []
        for _ in range(30):
            if board.is_game_over(claim_draw=True):
                break
            move = rng.choice(list(board.legal_moves))
            moves.append(move.uci())
            board.push(move)
        games.append({'root_fen': chess.STARTING_FEN, 'moves': moves})
    source.write_text(''.join(json.dumps(row) + '\n' for row in games))
    source.with_suffix('.json').write_text(json.dumps({
        'dataset': artifact_info(source), 'games': len(games),
        'source_urls': ['https://database.lichess.org/'], 'license': 'CC0'}))
    first, second = tmp_path / 'first.fen', tmp_path / 'second.fen'
    sample(source, first, count=5, seed=7)
    result = sample(source, second, count=5, seed=7)
    keys = lambda path: {' '.join(line.split()[:4]) for line in path.read_text().splitlines()}
    assert not keys(first) & keys(second)
    assert result['earlier_roots'] == [str(first)]
    with pytest.raises(ValueError, match='new output path'):
        sample(source, second, count=5, seed=7)
