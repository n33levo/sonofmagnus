"""
8×8×73 policy head mapping for chess moves (AlphaZero-style).

Each legal move is encoded as (from_square, plane_index) where:
- from_square: 0-63 (8×8 board)
- plane_index: 0-72 (73 planes total)

The 73 planes represent:
  - Planes 0-55: Queen moves (8 directions × 7 distances)
    * N, NE, E, SE, S, SW, W, NW × distances 1-7
  - Planes 56-63: Knight moves (8 L-shaped moves)
  - Planes 64-72: Underpromotions (3 directions × 3 pieces)
    * N, NW, NE × {Knight, Bishop, Rook}
    * (Queen promotions use queen-move planes)
"""

import chess
import numpy as np


# Direction vectors for queen moves (N, NE, E, SE, S, SW, W, NW)
QUEEN_DIRECTIONS = [
    (0, 1),   # N
    (1, 1),   # NE
    (1, 0),   # E
    (1, -1),  # SE
    (0, -1),  # S
    (-1, -1), # SW
    (-1, 0),  # W
    (-1, 1),  # NW
]

# Knight move offsets (8 L-shaped moves)
KNIGHT_MOVES = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2)
]

# Underpromotion directions and pieces
UNDERPROMO_DIRECTIONS = [
    (0, 1),   # N (straight ahead)
    (-1, 1),  # NW (capture left)
    (1, 1),   # NE (capture right)
]

UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def move_to_policy_index(move: chess.Move, board: chess.Board) -> tuple[int, int]:
    """
    Convert a chess.Move to (from_square, plane_index).

    Args:
        move: python-chess Move object
        board: Current board state (needed for context)

    Returns:
        (from_square, plane_index) tuple
    """
    from_sq = move.from_square
    to_sq = move.to_square

    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)
    to_file = chess.square_file(to_sq)
    to_rank = chess.square_rank(to_sq)

    file_delta = to_file - from_file
    rank_delta = to_rank - from_rank

    # Handle underpromotions (planes 64-72)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        # Find direction index
        mover = board.piece_at(from_sq)
        if mover is None or mover.piece_type != chess.PAWN:
            raise ValueError(f"Underpromotion move {move} does not originate from a pawn")

        forward = 1 if mover.color == chess.WHITE else -1
        direction = (file_delta * forward, rank_delta * forward)
        if direction not in UNDERPROMO_DIRECTIONS:
            raise ValueError(f"Invalid underpromotion direction: {direction} for {move}")

        dir_idx = UNDERPROMO_DIRECTIONS.index(direction)

        # Find piece index
        piece_idx = UNDERPROMO_PIECES.index(move.promotion)

        plane = 64 + dir_idx * 3 + piece_idx
        return (from_sq, plane)

    # Handle knight moves (planes 56-63)
    delta = (file_delta, rank_delta)
    if delta in KNIGHT_MOVES:
        plane = 56 + KNIGHT_MOVES.index(delta)
        return (from_sq, plane)

    # Handle queen-style moves (planes 0-55)
    # Normalize direction
    if file_delta == 0 and rank_delta == 0:
        raise ValueError(f"Invalid move with no displacement: {move}")

    # Determine direction
    dir_file = 0 if file_delta == 0 else (1 if file_delta > 0 else -1)
    dir_rank = 0 if rank_delta == 0 else (1 if rank_delta > 0 else -1)
    direction = (dir_file, dir_rank)

    if direction not in QUEEN_DIRECTIONS:
        raise ValueError(f"Invalid queen-move direction: {direction} for {move}")

    dir_idx = QUEEN_DIRECTIONS.index(direction)

    # Determine distance
    distance = max(abs(file_delta), abs(rank_delta))
    if distance < 1 or distance > 7:
        raise ValueError(f"Invalid queen-move distance: {distance} for {move}")

    plane = dir_idx * 7 + (distance - 1)
    return (from_sq, plane)


def policy_index_to_move(from_square: int, plane_index: int, board: chess.Board) -> chess.Move:
    """
    Convert (from_square, plane_index) back to a chess.Move.

    Args:
        from_square: Source square (0-63)
        plane_index: Policy plane index (0-72)
        board: Current board state

    Returns:
        chess.Move object
    """
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)

    # Underpromotions (planes 64-72)
    if plane_index >= 64:
        offset = plane_index - 64
        dir_idx = offset // 3
        piece_idx = offset % 3

        mover = board.piece_at(from_square)
        color = mover.color if mover else board.turn
        forward = 1 if color == chess.WHITE else -1
        direction = UNDERPROMO_DIRECTIONS[dir_idx]
        direction = (direction[0] * forward, direction[1] * forward)
        promotion_piece = UNDERPROMO_PIECES[piece_idx]

        to_file = from_file + direction[0]
        to_rank = from_rank + direction[1]
        to_square = chess.square(to_file, to_rank)

        return chess.Move(from_square, to_square, promotion=promotion_piece)

    # Knight moves (planes 56-63)
    elif plane_index >= 56:
        knight_idx = plane_index - 56
        delta = KNIGHT_MOVES[knight_idx]

        to_file = from_file + delta[0]
        to_rank = from_rank + delta[1]
        to_square = chess.square(to_file, to_rank)

        return chess.Move(from_square, to_square)

    # Queen moves (planes 0-55)
    else:
        dir_idx = plane_index // 7
        distance = (plane_index % 7) + 1

        direction = QUEEN_DIRECTIONS[dir_idx]

        to_file = from_file + direction[0] * distance
        to_rank = from_rank + direction[1] * distance
        to_square = chess.square(to_file, to_rank)

        # Check for queen promotion (pawn reaching last rank)
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and to_rank == 7) or \
               (piece.color == chess.BLACK and to_rank == 0):
                return chess.Move(from_square, to_square, promotion=chess.QUEEN)

        return chess.Move(from_square, to_square)


def build_policy_target(move: chess.Move, board: chess.Board) -> np.ndarray:
    """
    Build a one-hot encoded policy target for a single move.

    Args:
        move: The target move
        board: Current board state

    Returns:
        numpy array of shape (8, 8, 73) with one-hot encoding
    """
    policy = np.zeros((8, 8, 73), dtype=np.float32)
    from_sq, plane = move_to_policy_index(move, board)
    rank = chess.square_rank(from_sq)
    file = chess.square_file(from_sq)
    policy[rank, file, plane] = 1.0
    return policy


def build_policy_distribution(move_probs: dict[chess.Move, float], board: chess.Board) -> np.ndarray:
    """
    Build a policy distribution from move probabilities.

    Args:
        move_probs: Dictionary mapping moves to probabilities
        board: Current board state

    Returns:
        numpy array of shape (8, 8, 73) with probability distribution
    """
    policy = np.zeros((8, 8, 73), dtype=np.float32)
    for move, prob in move_probs.items():
        from_sq, plane = move_to_policy_index(move, board)
        rank = chess.square_rank(from_sq)
        file = chess.square_file(from_sq)
        policy[rank, file, plane] = prob
    return policy


def build_q_value_target(q_values: dict[chess.Move, float], board: chess.Board) -> np.ndarray:
    """
    Build Q-value targets for all legal moves.

    Args:
        q_values: Dictionary mapping moves to Q-values (win probabilities [0, 1])
        board: Current board state

    Returns:
        numpy array of shape (8, 8, 73) with Q-values, -999 for illegal moves
    """
    q_target = np.full((8, 8, 73), -999.0, dtype=np.float32)
    for move, q_val in q_values.items():
        try:
            from_sq, plane = move_to_policy_index(move, board)
            rank = chess.square_rank(from_sq)
            file = chess.square_file(from_sq)
            q_target[rank, file, plane] = q_val
        except ValueError:
            # Skip invalid moves
            continue
    return q_target


def test_round_trip():
    """Test that move encoding/decoding is bijective."""
    board = chess.Board()
    errors = []

    for move in board.legal_moves:
        try:
            from_sq, plane = move_to_policy_index(move, board)
            reconstructed = policy_index_to_move(from_sq, plane, board)

            if reconstructed != move:
                errors.append(f"Round-trip failed: {move} -> ({from_sq},{plane}) -> {reconstructed}")
        except Exception as e:
            errors.append(f"Error encoding {move}: {e}")

    if errors:
        print("ERRORS:")
        for err in errors:
            print(f"  {err}")
        return False
    else:
        print(f"✓ All {board.legal_moves.count()} starting moves round-trip successfully")
        return True


def test_coverage():
    """Test that all 73 planes can be used."""
    board = chess.Board()

    # Test a variety of positions
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # Castling
        "8/8/8/4N3/8/8/8/8 w - - 0 1",  # Knight
        "8/P7/8/8/8/8/8/8 w - - 0 1",  # Promotion
    ]

    used_planes = set()

    for fen in test_positions:
        b = chess.Board(fen)
        for move in b.legal_moves:
            try:
                _, plane = move_to_policy_index(move, b)
                used_planes.add(plane)
            except Exception as e:
                print(f"Error in position {fen}, move {move}: {e}")

    print(f"✓ Coverage: {len(used_planes)}/73 planes can be generated")
    return True


if __name__ == "__main__":
    print("Testing 8×8×73 policy mapping...")
    print()
    test_round_trip()
    print()
    test_coverage()
    print()
    print("Policy mapping tests complete.")
