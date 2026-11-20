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
    if file_delta and rank_delta and abs(file_delta) != abs(rank_delta):
        raise ValueError(f"Move is neither a ray nor a knight jump: {move}")
    if distance < 1 or distance > 7:
        raise ValueError(f"Invalid queen-move distance: {distance} for {move}")

    plane = dir_idx * 7 + (distance - 1)
    return (from_sq, plane)


def policy_index_to_move(from_square: int, plane_index: int, board: chess.Board) -> chess.Move:
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)

    if not 0 <= from_square < 64 or not 0 <= plane_index < 73:
        raise ValueError("Policy index out of range")

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
        if not 0 <= to_file < 8 or not 0 <= to_rank < 8:
            raise ValueError("Policy move leaves the board")
        to_square = chess.square(to_file, to_rank)

        return chess.Move(from_square, to_square, promotion=promotion_piece)

    # Knight moves (planes 56-63)
    elif plane_index >= 56:
        knight_idx = plane_index - 56
        delta = KNIGHT_MOVES[knight_idx]

        to_file = from_file + delta[0]
        to_rank = from_rank + delta[1]
        if not 0 <= to_file < 8 or not 0 <= to_rank < 8:
            raise ValueError("Policy move leaves the board")
        to_square = chess.square(to_file, to_rank)

        return chess.Move(from_square, to_square)

    # Queen moves (planes 0-55)
    else:
        dir_idx = plane_index // 7
        distance = (plane_index % 7) + 1

        direction = QUEEN_DIRECTIONS[dir_idx]

        to_file = from_file + direction[0] * distance
        to_rank = from_rank + direction[1] * distance
        if not 0 <= to_file < 8 or not 0 <= to_rank < 8:
            raise ValueError("Policy move leaves the board")
        to_square = chess.square(to_file, to_rank)

        # Check for queen promotion (pawn reaching last rank)
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and to_rank == 7) or \
               (piece.color == chess.BLACK and to_rank == 0):
                return chess.Move(from_square, to_square, promotion=chess.QUEEN)

        return chess.Move(from_square, to_square)


def build_policy_target(move: chess.Move, board: chess.Board) -> np.ndarray:
    policy = np.zeros((8, 8, 73), dtype=np.float32)
    if move not in board.legal_moves:
        raise ValueError(f"Illegal policy target: {move}")
    from_sq, plane = move_to_policy_index(move, board)
    rank = chess.square_rank(from_sq)
    file = chess.square_file(from_sq)
    policy[rank, file, plane] = 1.0
    return policy


def build_policy_distribution(move_probs: dict[chess.Move, float], board: chess.Board) -> np.ndarray:
    policy = np.zeros((8, 8, 73), dtype=np.float32)
    for move, prob in move_probs.items():
        if move not in board.legal_moves or not np.isfinite(prob) or prob < 0:
            raise ValueError(f"Invalid policy target: {move}: {prob}")
        from_sq, plane = move_to_policy_index(move, board)
        rank = chess.square_rank(from_sq)
        file = chess.square_file(from_sq)
        policy[rank, file, plane] = prob
    mass = policy.sum()
    if mass <= 0:
        raise ValueError("Empty policy distribution")
    policy /= mass
    return policy


def build_q_value_target(q_values: dict[chess.Move, float], board: chess.Board) -> np.ndarray:
    q_target = np.full((8, 8, 73), -999.0, dtype=np.float32)
    for move, q_val in q_values.items():
        if move not in board.legal_moves or not np.isfinite(q_val) or not 0 <= q_val <= 1:
            raise ValueError(f"Invalid Q target: {move}: {q_val}")
        from_sq, plane = move_to_policy_index(move, board)
        q_target[from_sq // 8, from_sq % 8, plane] = q_val
    return q_target
