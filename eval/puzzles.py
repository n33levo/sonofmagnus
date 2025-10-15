"""
Puzzle evaluation for chess model.

Tests tactical strength on Lichess puzzles.
"""

import argparse
import json
from collections import defaultdict

import chess
import torch

from play.runner import ChessRunner
from model import create_model_main, create_model_mini


def solve_puzzle(
    runner: ChessRunner,
    fen: str,
    solution_moves: list[str],
    max_attempts: int = 3,
) -> bool:
    """
    Attempt to solve a puzzle.

    Args:
        runner: ChessRunner instance
        fen: Starting FEN
        solution_moves: List of UCI moves (alternating player/opponent)
        max_attempts: Number of attempts per position

    Returns:
        True if puzzle solved correctly
    """
    board = chess.Board(fen)

    # Solution alternates: opponent's setup move, our response, opponent, our response...
    # For puzzles, we typically need to find the first move in the solution
    our_moves = solution_moves[::2]  # Even indices are "our" moves

    for i, our_move_uci in enumerate(our_moves):
        solution_move = chess.Move.from_uci(our_move_uci)

        # Try multiple times (temperature sampling)
        solved = False
        for attempt in range(max_attempts):
            predicted_move = runner.select_move(board, deterministic=(attempt == 0))

            if predicted_move == solution_move:
                solved = True
                break

        if not solved:
            return False

        # Make our move and opponent's response (if any)
        board.push(solution_move)

        # Make opponent's move if exists
        if i * 2 + 1 < len(solution_moves):
            opponent_move = chess.Move.from_uci(solution_moves[i * 2 + 1])
            board.push(opponent_move)

    return True


def evaluate_puzzles(
    model_path: str,
    puzzles_path: str,
    config: str = "main",
    device: str = "cpu",
    max_puzzles: int = None,
) -> dict:
    """
    Evaluate model on puzzle dataset.

    Args:
        model_path: Path to model checkpoint
        puzzles_path: Path to puzzles JSONL
        config: Model config
        device: Device
        max_puzzles: Max puzzles to evaluate

    Returns:
        Dictionary with accuracy by rating bucket
    """
    # Load model
    if config == "main":
        model = create_model_main()
    else:
        model = create_model_mini()

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    runner = ChessRunner(model, device=device)

    # Load puzzles
    puzzles = []
    with open(puzzles_path) as f:
        for i, line in enumerate(f):
            if max_puzzles and i >= max_puzzles:
                break
            puzzles.append(json.loads(line))

    print(f"Evaluating on {len(puzzles)} puzzles...")

    # Evaluate
    by_rating = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, puzzle in enumerate(puzzles):
        fen = puzzle["fen"]
        solution = puzzle["moves"]  # List of UCI moves
        rating = puzzle.get("rating", 1500)

        # Bucket by rating
        rating_bucket = (rating // 100) * 100

        solved = solve_puzzle(runner, fen, solution)

        by_rating[rating_bucket]["total"] += 1
        if solved:
            by_rating[rating_bucket]["correct"] += 1

        if (i + 1) % 100 == 0:
            print(f"  Evaluated {i + 1}/{len(puzzles)} puzzles...")

    # Compute accuracies
    results = {}
    total_correct = 0
    total_count = 0

    for bucket in sorted(by_rating.keys()):
        stats = by_rating[bucket]
        accuracy = stats["correct"] / stats["total"]
        results[bucket] = {
            "accuracy": accuracy,
            "correct": stats["correct"],
            "total": stats["total"],
        }
        total_correct += stats["correct"]
        total_count += stats["total"]

    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0
    results["overall"] = {
        "accuracy": overall_accuracy,
        "correct": total_correct,
        "total": total_count,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on puzzles")
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--puzzles", type=str, required=True, help="Puzzles JSONL")
    parser.add_argument("--config", type=str, choices=["main", "mini"], default="main")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max", type=int, help="Max puzzles to evaluate")
    args = parser.parse_args()

    results = evaluate_puzzles(
        model_path=args.ckpt,
        puzzles_path=args.puzzles,
        config=args.config,
        device=args.device,
        max_puzzles=args.max,
    )

    print("\nPuzzle Results:")
    print(f"Overall: {results['overall']['accuracy']:.2%} ({results['overall']['correct']}/{results['overall']['total']})")
    print("\nBy rating:")
    for bucket in sorted(k for k in results.keys() if k != "overall"):
        stats = results[bucket]
        print(f"  {bucket}-{bucket+99}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    main()
