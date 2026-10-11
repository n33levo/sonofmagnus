import chess
import numpy as np


def _policy_moves():
    moves = [chess.Move(a, b) for a in range(64) for b in range(64) if a != b and
             (a % 8 == b % 8 or a // 8 == b // 8 or abs(a % 8 - b % 8) == abs(a // 8 - b // 8)
              or sorted((abs(a % 8 - b % 8), abs(a // 8 - b // 8))) == [1, 2])]
    moves += [chess.Move(a, b, promotion=p) for a in range(48, 56) for b in range(56, 64)
              if abs(a % 8 - b % 8) <= 1 for p in (chess.QUEEN, chess.ROOK, chess.BISHOP)]
    return {move: index for index, move in enumerate(moves)}


POLICY_INDICES = _policy_moves()


def policy_index(move, board):
    start, end, promotion = move.from_square, move.to_square, move.promotion
    if board.is_castling(move):
        end = chess.square(7 if end > start else 0, start // 8)
    if not board.turn:
        start, end = chess.square_mirror(start), chess.square_mirror(end)
    return POLICY_INDICES[chess.Move(start, end, promotion=None if promotion == chess.KNIGHT else promotion)]


def encode_history(board):
    """Classical LC0 planes, with the current mover's colours and rank orientation."""
    if board.chess960:
        raise ValueError('Classical LC0 encoding supports standard chess')
    planes = np.zeros((112, 64), dtype=np.float32)
    turn = board.turn
    squares = np.arange(64, dtype=np.uint64)
    cursor = board.copy()
    missing = False
    for frame in range(8):
        if missing and cursor.board_fen() == chess.STARTING_BOARD_FEN and cursor.turn and cursor.castling_rights == chess.BB_CORNERS:
            break
        masks = [cursor.pieces_mask(piece, color) for color in (turn, not turn) for piece in chess.PIECE_TYPES]
        if missing and cursor.has_legal_en_passant():
            mover = not cursor.turn
            end = cursor.ep_square + (8 if mover else -8)
            start = cursor.ep_square - (8 if mover else -8)
            index = 0 if mover == turn else 6
            masks[index] ^= (1 << end) | (1 << start)
        bits = (np.asarray(masks, dtype=np.uint64)[:, None] >> squares) & 1
        if not turn:
            bits = bits.reshape(12, 8, 8)[:, ::-1].reshape(12, 64)
        planes[frame * 13:frame * 13 + 12] = bits
        planes[frame * 13 + 12] = cursor.is_repetition(2)
        if cursor.move_stack:
            cursor.pop()
        else:
            missing = True
    for offset, color in ((104, turn), (106, not turn)):
        planes[offset] = board.has_queenside_castling_rights(color)
        planes[offset + 1] = board.has_kingside_castling_rights(color)
    planes[108] = not turn
    planes[109] = board.halfmove_clock
    planes[111] = 1
    return planes.reshape(112, 8, 8)
