import chess


def record_board(row):
    if 'history' not in row:
        return chess.Board(row['fen'])
    board = chess.Board(row.get('root_fen', row['fen']))
    for move in row.get('history', []):
        board.push_uci(move)
    if board.fen() != chess.Board(row['fen']).fen():
        raise ValueError('Stored position and move history disagree')
    return board
