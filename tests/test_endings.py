"""
Ending conditions tests (Gate G1 extended).

Ensures correct handling of:
- Stalemate
- Fifty-move rule
- Threefold repetition
- Promotions
- Checkmate
"""

import chess
import numpy as np
import torch

from chessio.encode import encode_board
from chessio.mask import sample_legal_move
from model import create_model_mini


def test_stalemate_detection():
    """Test that stalemate is correctly detected."""
    print("Testing stalemate detection...")

    # Famous stalemate position
    fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    board = chess.Board(fen)

    assert board.is_stalemate(), "Should detect stalemate"
    assert board.is_game_over(), "Game should be over"
    assert not board.is_checkmate(), "Should not be checkmate"

    print("✓ Stalemate correctly detected")


def test_fifty_move_rule():
    """Test fifty-move rule detection."""
    print("Testing fifty-move rule...")

    board = chess.Board()

    # Make 50 moves without capture or pawn move
    board.halfmove_clock = 100  # 50 full moves

    assert board.can_claim_fifty_moves(), "Should allow fifty-move claim"

    print("✓ Fifty-move rule correctly detected")


def test_threefold_repetition():
    """Test threefold repetition detection."""
    print("Testing threefold repetition...")

    board = chess.Board()

    # Repeat position 3 times by moving knights back and forth
    moves = [
        chess.Move.from_uci("g1f3"),
        chess.Move.from_uci("g8f6"),
        chess.Move.from_uci("f3g1"),
        chess.Move.from_uci("f6g8"),
    ]

    # First occurrence
    for move in moves:
        board.push(move)

    # Second occurrence
    for move in moves:
        board.push(move)

    # Should be able to claim threefold
    assert board.can_claim_threefold_repetition(), "Should detect threefold repetition"

    print("✓ Threefold repetition correctly detected")


def test_promotions():
    """Test that pawn promotions work correctly."""
    print("Testing pawn promotions...")

    # White pawn about to promote
    fen = "8/P7/8/8/8/8/8/8 w - - 0 1"
    board = chess.Board(fen)

    # Should have 4 promotion moves (Q, R, B, N) going forward
    promo_moves = [m for m in board.legal_moves if m.promotion]
    assert len(promo_moves) == 4, f"Should have 4 promotions, got {len(promo_moves)}"

    # Test each promotion type
    for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        test_board = board.copy()
        move = chess.Move(chess.A7, chess.A8, promotion=piece_type)
        assert move in test_board.legal_moves
        test_board.push(move)
        assert test_board.piece_at(chess.A8).piece_type == piece_type

    print("✓ All promotion types work correctly")


def test_checkmate_detection():
    """Test checkmate detection."""
    print("Testing checkmate detection...")

    # Scholar's mate position
    fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    board = chess.Board(fen)

    assert board.is_checkmate(), "Should detect checkmate"
    assert board.is_game_over(), "Game should be over"
    assert not board.is_stalemate(), "Should not be stalemate"

    print("✓ Checkmate correctly detected")


def test_model_handles_endings():
    """Test that model can handle various ending scenarios."""
    print("Testing model with ending scenarios...")

    model = create_model_mini()
    model.eval()

    test_positions = [
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "stalemate"),
        ("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4", "checkmate"),
        ("8/P7/8/8/8/8/8/8 w - - 0 1", "promotion"),
    ]

    with torch.no_grad():
        for fen, scenario in test_positions:
            board = chess.Board(fen)

            if board.is_game_over():
                # Model shouldn't be asked to move in game-over positions
                continue

            if board.legal_moves.count() == 0:
                continue

            # Generate move
            board_tensor = encode_board(board)
            board_tensor = torch.from_numpy(board_tensor).unsqueeze(0)

            policy_logits, _ = model(board_tensor)
            policy_logits = policy_logits[0].cpu().numpy()

            move = sample_legal_move(policy_logits, board)
            assert move in board.legal_moves, f"Illegal move in {scenario} scenario"

    print("✓ Model correctly handles ending scenarios")


def run_all_tests():
    """Run all ending tests."""
    print("="*60)
    print("ENDING CONDITIONS TESTS")
    print("="*60)
    print()

    test_stalemate_detection()
    print()

    test_fifty_move_rule()
    print()

    test_threefold_repetition()
    print()

    test_promotions()
    print()

    test_checkmate_detection()
    print()

    test_model_handles_endings()
    print()

    print("="*60)
    print("✓ ALL ENDING TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
