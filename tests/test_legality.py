"""
Legality tests (Gate G1).

Ensures:
- Zero illegal moves selected across random boards
- Masked softmax sums to 1.0
- All legal moves are representable
"""

import chess
import numpy as np
import torch

from chessio.encode import encode_board
from chessio.mask import build_legal_mask, apply_legal_mask, sample_legal_move, argmax_legal_move
from chessio.policy_map import move_to_policy_index, policy_index_to_move
from model import create_model_mini


def test_mask_coverage():
    """Test that legal mask correctly identifies all legal moves."""
    print("Testing legal mask coverage...")

    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # Castling
        "8/P7/8/8/8/8/8/8 w - - 0 1",  # Promotion
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # En passant
    ]

    for fen in test_fens:
        board = chess.Board(fen)
        mask = build_legal_mask(board)

        # Count masked moves
        num_legal_from_board = board.legal_moves.count()
        num_legal_from_mask = int(mask.sum())

        assert num_legal_from_board == num_legal_from_mask, \
            f"Mismatch for {fen}: board={num_legal_from_board}, mask={num_legal_from_mask}"

    print(f"✓ Passed coverage test on {len(test_fens)} positions")


def test_masked_softmax_sums_to_one():
    """Test that masked softmax sums to 1.0."""
    print("Testing masked softmax normalization...")

    board = chess.Board()
    mask = build_legal_mask(board)

    # Random logits
    logits = np.random.randn(8, 8, 73).astype(np.float32)
    masked_logits = apply_legal_mask(logits, mask)

    # Softmax
    flat = masked_logits.reshape(-1)
    flat = flat - np.max(flat)
    exp_flat = np.exp(flat)
    probs = exp_flat / np.sum(exp_flat)

    total_prob = np.sum(probs)
    assert abs(total_prob - 1.0) < 1e-6, f"Softmax doesn't sum to 1: {total_prob}"

    print(f"✓ Masked softmax sums to 1.0 (total={total_prob:.9f})")


def test_no_illegal_moves_random_boards(num_boards: int = 1000):
    """Test that sampling never produces illegal moves on random positions."""
    print(f"Testing illegal move rate on {num_boards} random boards...")

    illegal_count = 0

    for _ in range(num_boards):
        # Random position (play random moves from start)
        board = chess.Board()
        for _ in range(np.random.randint(5, 30)):
            if board.is_game_over():
                break
            legal_moves = list(board.legal_moves)
            move = np.random.choice(legal_moves)
            board.push(move)

        if board.is_game_over():
            continue

        # Random logits
        logits = np.random.randn(8, 8, 73).astype(np.float32)

        # Sample move
        try:
            move = sample_legal_move(logits, board, temperature=1.0)
            if move not in board.legal_moves:
                illegal_count += 1
        except Exception as e:
            print(f"Error sampling move: {e}")
            illegal_count += 1

    illegal_rate = illegal_count / num_boards
    print(f"Illegal move rate: {illegal_rate:.2%} ({illegal_count}/{num_boards})")

    assert illegal_count == 0, f"Found {illegal_count} illegal moves!"
    print(f"✓ Zero illegal moves across {num_boards} random boards")


def test_no_illegal_moves_with_model():
    """Test that model never produces illegal moves."""
    print("Testing model-generated moves...")

    model = create_model_mini()
    model.eval()

    num_positions = 100
    illegal_count = 0

    with torch.no_grad():
        for _ in range(num_positions):
            # Random position
            board = chess.Board()
            for _ in range(np.random.randint(5, 20)):
                if board.is_game_over():
                    break
                legal_moves = list(board.legal_moves)
                move = np.random.choice(legal_moves)
                board.push(move)

            if board.is_game_over():
                continue

            # Get model prediction
            board_tensor = encode_board(board)
            board_tensor = torch.from_numpy(board_tensor).unsqueeze(0)

            policy_logits, _ = model(board_tensor)
            policy_logits = policy_logits[0].cpu().numpy()

            # Sample move
            try:
                move = sample_legal_move(policy_logits, board)
                if move not in board.legal_moves:
                    illegal_count += 1
            except Exception:
                illegal_count += 1

    assert illegal_count == 0, f"Model produced {illegal_count} illegal moves!"
    print(f"✓ Model produced zero illegal moves across {num_positions} positions")


def run_all_tests():
    """Run all legality tests."""
    print("="*60)
    print("LEGALITY TESTS (Gate G1)")
    print("="*60)
    print()

    test_mask_coverage()
    print()

    test_masked_softmax_sums_to_one()
    print()

    test_no_illegal_moves_random_boards(num_boards=10000)
    print()

    test_no_illegal_moves_with_model()
    print()

    print("="*60)
    print("✓ ALL LEGALITY TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
