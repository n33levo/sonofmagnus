"""
Board-to-tensor encoding for chess positions.

Encodes a chess.Board into a multi-channel tensor representation:
- 12 piece planes (own/opponent × {K,Q,R,B,N,P})
- 1 side-to-move plane
- 4 castling rights planes (K-side/Q-side for each player)
- 1 halfmove clock plane (normalized)
- Optional: history frames for temporal context
"""

import numpy as np
import chess


def encode_board(board: chess.Board, history_frames: int = 0) -> np.ndarray:
    """
    Encode a chess board into a tensor representation.

    Args:
        board: python-chess Board object
        history_frames: Number of historical positions to include (not implemented yet)

    Returns:
        numpy array of shape (C, 8, 8) where C = 18 (base encoding)
        Channels:
          0-5: Own pieces (K, Q, R, B, N, P)
          6-11: Opponent pieces (K, Q, R, B, N, P)
          12: Side to move (1 if white, 0 if black)
          13: White kingside castling
          14: White queenside castling
          15: Black kingside castling
          16: Black queenside castling
          17: Halfmove clock (normalized to [0,1] by dividing by 100)
    """
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
    """
    Encode a batch of chess boards.

    Args:
        boards: List of python-chess Board objects
        history_frames: Number of historical positions to include

    Returns:
        numpy array of shape (N, C, 8, 8)
    """
    return np.stack([encode_board(b, history_frames) for b in boards], axis=0)


def flip_board_vertical(tensor: np.ndarray) -> np.ndarray:
    """
    Flip board representation vertically (rank symmetry).
    Used for data augmentation.

    Args:
        tensor: Board encoding of shape (C, 8, 8)

    Returns:
        Vertically flipped tensor
    """
    return np.flip(tensor, axis=1).copy()


def flip_board_horizontal(tensor: np.ndarray) -> np.ndarray:
    """
    Flip board representation horizontally (file symmetry).
    Used for data augmentation.

    Args:
        tensor: Board encoding of shape (C, 8, 8)

    Returns:
        Horizontally flipped tensor
    """
    return np.flip(tensor, axis=2).copy()


def swap_colors(tensor: np.ndarray) -> np.ndarray:
    """
    Swap piece colors and flip board for color-swap augmentation.

    Args:
        tensor: Board encoding of shape (C, 8, 8)

    Returns:
        Color-swapped and flipped tensor
    """
    result = tensor.copy()

    # Swap own pieces (0-5) with opponent pieces (6-11)
    result[0:6], result[6:12] = tensor[6:12].copy(), tensor[0:6].copy()

    # Flip the board vertically (since we're viewing from the other side)
    result = np.flip(result, axis=1).copy()

    # Flip side to move
    result[12] = 1.0 - result[12]

    # Swap castling rights
    result[13], result[15] = result[15].copy(), result[13].copy()  # Kingside
    result[14], result[16] = result[16].copy(), result[14].copy()  # Queenside

    return result


if __name__ == "__main__":
    # Quick sanity check
    board = chess.Board()
    tensor = encode_board(board)
    print(f"Encoded starting position: shape={tensor.shape}")
    print(f"White pawns on rank 1 (from white's perspective): {tensor[5, 1, :].sum()} (expect 8)")
    print(f"Black pawns: {tensor[11, 6, :].sum()} (expect 8)")
    print(f"Side to move (white=1): {tensor[12, 0, 0]}")
    print(f"All castling rights: {tensor[13:17, 0, 0]}")

    # Test augmentation
    flipped = swap_colors(tensor)
    print(f"\nAfter color swap:")
    print(f"Now opponent pawns (was white): {flipped[11, 6, :].sum()}")
    print(f"Now own pawns (was black): {flipped[5, 1, :].sum()}")
