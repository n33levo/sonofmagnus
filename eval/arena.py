"""
Arena for round-robin model evaluation.

Plays multiple games between model snapshots to measure relative strength.
"""

import argparse
import time
from collections import defaultdict

import torch

from play.runner import ChessRunner
from model import create_model_main, create_model_mini


def play_match(
    runner_a: ChessRunner,
    runner_b: ChessRunner,
    num_games: int = 100,
    time_control: float = 0.2,
) -> dict:
    """
    Play a match between two runners.

    Args:
        runner_a: First runner
        runner_b: Second runner
        num_games: Number of games (half as white, half as black)
        time_control: Time control (not enforced, just for reference)

    Returns:
        Dictionary with match results
    """
    results = {"wins_a": 0, "wins_b": 0, "draws": 0}
    latencies_a = []
    latencies_b = []

    print(f"Playing {num_games} games...")

    for game_idx in range(num_games):
        # Alternate colors
        if game_idx % 2 == 0:
            white_runner = runner_a
            black_runner = runner_b
            a_is_white = True
        else:
            white_runner = runner_b
            black_runner = runner_a
            a_is_white = False

        # Play game
        start_time = time.time()
        game_result = white_runner.play_game(
            opponent_runner=black_runner,
            max_moves=200,
            verbose=False,
        )
        elapsed = time.time() - start_time

        # Record result
        result = game_result["result"]
        if result == "1-0":
            if a_is_white:
                results["wins_a"] += 1
            else:
                results["wins_b"] += 1
        elif result == "0-1":
            if a_is_white:
                results["wins_b"] += 1
            else:
                results["wins_a"] += 1
        else:
            results["draws"] += 1

        # Estimate latencies (rough)
        moves_per_player = game_result["num_moves"] / 2
        latency_per_move = elapsed / game_result["num_moves"] if game_result["num_moves"] > 0 else 0
        latencies_a.append(latency_per_move)
        latencies_b.append(latency_per_move)

        if (game_idx + 1) % 10 == 0:
            print(f"  Completed {game_idx + 1}/{num_games} games")

    # Compute statistics
    total_games = results["wins_a"] + results["wins_b"] + results["draws"]
    results["win_rate_a"] = results["wins_a"] / total_games if total_games > 0 else 0.0
    results["win_rate_b"] = results["wins_b"] / total_games if total_games > 0 else 0.0
    results["draw_rate"] = results["draws"] / total_games if total_games > 0 else 0.0

    import numpy as np
    results["avg_latency_ms"] = np.mean(latencies_a) * 1000 if latencies_a else 0.0

    return results


def main():
    parser = argparse.ArgumentParser(description="Arena evaluation")
    parser.add_argument("--a", type=str, required=True, help="Model A checkpoint")
    parser.add_argument("--b", type=str, required=True, help="Model B checkpoint")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument("--config", type=str, choices=["main", "mini"], default="main")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tc", type=str, default="0.2", help="Time control (reference only)")
    args = parser.parse_args()

    # Load models
    print(f"Loading model A: {args.a}")
    if args.config == "main":
        model_a = create_model_main()
    else:
        model_a = create_model_mini()

    checkpoint_a = torch.load(args.a, map_location=args.device)
    model_a.load_state_dict(checkpoint_a["model_state_dict"])
    runner_a = ChessRunner(model_a, device=args.device)

    print(f"Loading model B: {args.b}")
    if args.config == "main":
        model_b = create_model_main()
    else:
        model_b = create_model_mini()

    checkpoint_b = torch.load(args.b, map_location=args.device)
    model_b.load_state_dict(checkpoint_b["model_state_dict"])
    runner_b = ChessRunner(model_b, device=args.device)

    # Play match
    results = play_match(runner_a, runner_b, num_games=args.games)

    # Print results
    print("\n" + "="*50)
    print("Arena Results:")
    print("="*50)
    print(f"Model A: {args.a}")
    print(f"Model B: {args.b}")
    print(f"\nGames played: {args.games}")
    print(f"\nModel A: {results['wins_a']} wins ({results['win_rate_a']:.1%})")
    print(f"Model B: {results['wins_b']} wins ({results['win_rate_b']:.1%})")
    print(f"Draws: {results['draws']} ({results['draw_rate']:.1%})")
    print(f"\nAvg latency: {results['avg_latency_ms']:.1f}ms/move")
    print("="*50)


if __name__ == "__main__":
    main()
