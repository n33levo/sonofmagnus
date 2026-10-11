import chess
import json
from pathlib import Path


def record_board(row):
    if 'history' not in row:
        return chess.Board(row['fen'])
    board = chess.Board(row.get('root_fen', row['fen']))
    for move in row.get('history', []):
        board.push_uci(move)
    if board.fen() != chess.Board(row['fen']).fen():
        raise ValueError('Stored position and move history disagree')
    return board


def holdout_positions(paths):
    blocked = set()
    for path in paths:
        with Path(path).open() as source:
            for line in source:
                row = json.loads(line)
                board = record_board(row)
                blocked.add(' '.join(board.fen().split()[:4]))
                for move in row.get('moves', []):
                    board.push_uci(move)
                    blocked.add(' '.join(board.fen().split()[:4]))
    return blocked
