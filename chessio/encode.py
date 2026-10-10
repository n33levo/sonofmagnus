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

    # Encode pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        rank = chess.square_rank(square)
        file = chess.square_file(square)

        # Determine piece channel offset
        piece_idx = piece_types.index(piece.piece_type)

        # Own pieces vs opponent pieces
        if piece.color == board.turn:
            # Own pieces: channels 0-5
            channel = piece_idx
        else:
            # Opponent pieces: channels 6-11
            channel = piece_idx + 6

        tensor[channel, rank, file] = 1.0

    # Channel 12: Side to move
    tensor[12, :, :] = 1.0 if is_white_turn else 0.0

    # Channels 13-16: Castling rights
    tensor[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # Channel 17: Halfmove clock (normalized)
    tensor[17, :, :] = min(board.halfmove_clock / 100.0, 1.0)

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
