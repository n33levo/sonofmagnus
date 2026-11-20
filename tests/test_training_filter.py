import json
from collections import Counter

import chess

from chessio.records import holdout_positions
from train.dataset import ChessPositionDataset


def test_training_excludes_the_whole_puzzle_line_and_clock_aliases(tmp_path):
    puzzle = tmp_path / 'puzzles.jsonl'
    puzzle.write_text(json.dumps({'fen': chess.STARTING_FEN, 'moves': ['e2e4', 'e7e5']}) + '\n')
    blocked = holdout_positions([puzzle])
    board = chess.Board()
    rows = [{'fen': board.fen(), 'move': 'e2e4'}]
    board.push_uci('e2e4')
    rows.append({'fen': board.fen(), 'move': 'e7e5'})
    board.push_uci('e7e5')
    rows.append({'fen': board.fen(), 'move': 'g1f3'})
    board.halfmove_clock, board.fullmove_number = 12, 42
    rows.append({'fen': board.fen(), 'move': 'g1f3'})
    board.push_uci('g1f3')
    rows.append({'fen': board.fen(), 'move': 'b8c6'})
    data = tmp_path / 'train.jsonl'
    data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    dataset = ChessPositionDataset(data, exclude=blocked)
    assert dataset.excluded == 4 and len(dataset) == 1
    assert dataset.read_row(0)['move'] == 'b8c6'


def test_tier_sampling_is_seeded_and_keeps_engine_rows(tmp_path):
    path = tmp_path / 'human.jsonl'
    rows = [{'fen': chess.STARTING_FEN, 'move': 'e2e4', 'weight': tier} for tier in (1., 1.5, 2., 3.) for _ in range(1000)]
    rows += [{'fen': chess.STARTING_FEN, 'move_probs': {'e2e4': 1.}, 'weight': 1.} for _ in range(100)]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    first = ChessPositionDataset(path, tier_sampling=True, seed=7)
    second = ChessPositionDataset(path, tier_sampling=True, seed=7)
    assert first.offsets == second.offsets
    counts = Counter(first.read_row(i)['weight'] for i in range(len(first)))
    assert counts[2.] == counts[3.] == 1000
    assert 750 < counts[1.5] < 850 and 550 < counts[1.] < 650
    assert sum('move_probs' in first.read_row(i) for i in range(len(first))) == 100
