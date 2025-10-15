"""
Legality masking for policy outputs.

Ensures the neural network only selects legal moves by:
1. Building a binary mask from python-chess legal moves
2. Zeroing illegal logits
3. Renormalizing the probability distribution
"""

import chess
import numpy as np
import torch
from chessio.policy_map import move_to_policy_index


def build_legal_mask(board: chess.Board) -> np.ndarray:
    """
    Build a binary mask indicating legal moves.

    Args:
        board: Current chess position

    Returns:
        numpy array of shape (8, 8, 73) where 1.0 = legal, 0.0 = illegal
    """
    mask = np.zeros((8, 8, 73), dtype=np.float32)

    for move in board.legal_moves:
        try:
            from_sq, plane = move_to_policy_index(move, board)
            rank = chess.square_rank(from_sq)
            file = chess.square_file(from_sq)
            mask[rank, file, plane] = 1.0
        except Exception as e:
            # This should never happen if policy_map is correct
            print(f"Warning: failed to encode legal move {move}: {e}")

    return mask


def build_legal_mask_torch(board: chess.Board, device: str = "cpu") -> torch.Tensor:
    """
    Build a legal mask as a PyTorch tensor.

    Args:
        board: Current chess position
        device: torch device (cpu/cuda)

    Returns:
        torch.Tensor of shape (8, 8, 73)
    """
    mask = build_legal_mask(board)
    return torch.from_numpy(mask).to(device)


def apply_legal_mask(logits: np.ndarray, mask: np.ndarray, large_negative: float = -1e9) -> np.ndarray:
    """
    Apply legality mask to policy logits.

    Args:
        logits: Raw policy logits of shape (8, 8, 73)
        mask: Binary mask of shape (8, 8, 73)
        large_negative: Value to set for illegal moves

    Returns:
        Masked logits (illegal moves set to large_negative)
    """
    masked_logits = logits.copy()
    masked_logits[mask == 0.0] = large_negative
    return masked_logits


def apply_legal_mask_torch(logits: torch.Tensor, mask: torch.Tensor, large_negative: float = -1e9) -> torch.Tensor:
    """
    Apply legality mask to policy logits (PyTorch version).

    Args:
        logits: Raw policy logits of shape (8, 8, 73) or (B, 8, 8, 73)
        mask: Binary mask of shape (8, 8, 73) or (B, 8, 8, 73)
        large_negative: Value to set for illegal moves

    Returns:
        Masked logits
    """
    masked_logits = logits.clone()
    masked_logits[mask == 0.0] = large_negative
    return masked_logits


def sample_legal_move(policy_logits: np.ndarray, board: chess.Board, temperature: float = 1.0) -> chess.Move:
    """
    Sample a legal move from masked policy logits.

    Args:
        policy_logits: Raw policy logits of shape (8, 8, 73)
        board: Current board state
        temperature: Sampling temperature (lower = more deterministic)

    Returns:
        Sampled legal chess.Move
    """
    # Build and apply mask
    mask = build_legal_mask(board)
    masked_logits = apply_legal_mask(policy_logits, mask)

    # Apply temperature
    if temperature != 1.0:
        masked_logits = masked_logits / temperature

    # Flatten and softmax
    flat_logits = masked_logits.reshape(-1)
    flat_logits = flat_logits - np.max(flat_logits)  # Numerical stability
    exp_logits = np.exp(flat_logits)
    probs = exp_logits / np.sum(exp_logits)

    # Sample index
    idx = np.random.choice(len(probs), p=probs)

    # Convert flat index back to (rank, file, plane)
    plane = idx % 73
    idx //= 73
    file = idx % 8
    rank = idx // 8

    from_sq = chess.square(file, rank)

    # Convert to move
    from chessio.policy_map import policy_index_to_move
    move = policy_index_to_move(from_sq, plane, board)

    return move


def argmax_legal_move(policy_logits: np.ndarray, board: chess.Board) -> chess.Move:
    """
    Select the highest-probability legal move (greedy).

    Args:
        policy_logits: Raw policy logits of shape (8, 8, 73)
        board: Current board state

    Returns:
        Best legal chess.Move according to logits
    """
    # Build and apply mask
    mask = build_legal_mask(board)
    masked_logits = apply_legal_mask(policy_logits, mask)

    # Find argmax
    flat_logits = masked_logits.reshape(-1)
    idx = np.argmax(flat_logits)

    # Convert flat index back to (rank, file, plane)
    plane = idx % 73
    idx //= 73
    file = idx % 8
    rank = idx // 8

    from_sq = chess.square(file, rank)

    # Convert to move
    from chessio.policy_map import policy_index_to_move
    move = policy_index_to_move(from_sq, plane, board)

    return move


def get_legal_move_probs(policy_logits: np.ndarray, board: chess.Board) -> dict[chess.Move, float]:
    """
    Get probability distribution over all legal moves.

    Args:
        policy_logits: Raw policy logits of shape (8, 8, 73)
        board: Current board state

    Returns:
        Dictionary mapping legal moves to probabilities
    """
    mask = build_legal_mask(board)
    masked_logits = apply_legal_mask(policy_logits, mask)

    # Softmax
    flat_logits = masked_logits.reshape(-1)
    flat_logits = flat_logits - np.max(flat_logits)
    exp_logits = np.exp(flat_logits)
    probs = exp_logits / np.sum(exp_logits)

    # Build move -> prob mapping
    move_probs = {}
    from chessio.policy_map import move_to_policy_index

    for move in board.legal_moves:
        from_sq, plane = move_to_policy_index(move, board)
        rank = chess.square_rank(from_sq)
        file = chess.square_file(from_sq)

        flat_idx = rank * 8 * 73 + file * 73 + plane
        move_probs[move] = float(probs[flat_idx])

    return move_probs


if __name__ == "__main__":
    # Sanity tests
    board = chess.Board()

    print("Testing legal mask...")
    mask = build_legal_mask(board)
    num_legal = int(mask.sum())
    num_legal_moves = board.legal_moves.count()

    print(f"Legal moves from board: {num_legal_moves}")
    print(f"Legal moves from mask: {num_legal}")
    assert num_legal == num_legal_moves, "Mismatch in legal move count!"

    print("\nTesting masked sampling...")
    fake_logits = np.random.randn(8, 8, 73).astype(np.float32)
    move = sample_legal_move(fake_logits, board, temperature=1.0)
    print(f"Sampled move: {move}")
    assert move in board.legal_moves, "Sampled illegal move!"

    print("\nTesting argmax selection...")
    move = argmax_legal_move(fake_logits, board)
    print(f"Argmax move: {move}")
    assert move in board.legal_moves, "Selected illegal move!"

    print("\nTesting probability distribution...")
    move_probs = get_legal_move_probs(fake_logits, board)
    total_prob = sum(move_probs.values())
    print(f"Total probability: {total_prob:.6f}")
    assert abs(total_prob - 1.0) < 1e-5, "Probabilities don't sum to 1!"

    print("\n✓ All mask tests passed!")
