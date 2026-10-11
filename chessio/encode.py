import numpy as np
import chess


def encode_board(board: chess.Board, history_frames: int = 0) -> np.ndarray:
    if history_frames:
        raise ValueError('The 18-plane encoding does not include history frames')
    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    # Piece type mapping
    piece_types = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]

    # Determine who is to move
    is_white_turn = board.turn == chess.WHITE

    masks = [board.pieces_mask(piece, color) for color in (board.turn, not board.turn) for piece in piece_types]
    tensor[:12] = ((np.asarray(masks, dtype=np.uint64)[:, None] >> np.arange(64, dtype=np.uint64)) & 1).reshape(12, 8, 8)

    # Channel 12: Side to move
    tensor[12, :, :] = 1.0 if is_white_turn else 0.0

    # Channels 13-16: Castling rights
    tensor[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # Channel 17: Halfmove clock (normalized)
    tensor[17, :, :] = min(board.halfmove_clock / 100.0, 1.0)
    # One spatial marker carries en passant without adding a channel; the clock is the plane's minimum.
    if board.has_legal_en_passant():
        tensor[17, board.ep_square // 8, board.ep_square % 8] += 1.

    return tensor


def encode_board_batch(boards: list[chess.Board], history_frames: int = 0) -> np.ndarray:
    return np.stack([encode_board(b, history_frames) for b in boards], axis=0)


def flip_board_vertical(tensor: np.ndarray) -> np.ndarray:
    return np.flip(tensor, axis=1).copy()


def flip_board_horizontal(tensor: np.ndarray) -> np.ndarray:
    return np.flip(tensor, axis=2).copy()


def swap_colors(tensor: np.ndarray) -> np.ndarray:
    result = tensor.copy()

    # Flip the board vertically (since we're viewing from the other side)
    result = np.flip(result, axis=1).copy()

    # Flip side to move
    result[12] = 1.0 - result[12]

    # Swap castling rights
    result[13], result[15] = result[15].copy(), result[13].copy()  # Kingside
    result[14], result[16] = result[16].copy(), result[14].copy()  # Queenside

    return result
