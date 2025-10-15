"""
Searchless self-play for DAgger-style data collection.

Plays games using current policy and logs visited positions (FENs)
for later relabeling with teacher (engine).
"""

import argparse
import json
from pathlib import Path

import chess
import torch

from play.runner import ChessRunner
from model import create_model_main, create_model_mini


def play_selfplay_game(
    runner: ChessRunner,
    max_moves: int = 200,
) -> list[str]:
    """
    Play a self-play game and collect visited FENs.

    Args:
        runner: ChessRunner instance
        max_moves: Maximum moves before draw

    Returns:
        List of FEN strings visited during the game
    """
    board = chess.Board()
    fens = []

    while not board.is_game_over() and len(fens) < max_moves:
        # Record position before move
        fens.append(board.fen())

        # Select and make move
        move = runner.select_move(board)
        board.push(move)

    return fens


def run_selfplay(
    model_path: str,
    output_path: str,
    num_games: int = 100,
    config: str = "main",
    device: str = "cpu",
    max_moves: int = 200,
):
    """
    Run self-play games and save visited positions.

    Args:
        model_path: Path to trained model checkpoint
        output_path: Output JSONL file for FENs
        num_games: Number of games to play
        config: Model config (main/mini)
        device: Device to use
        max_moves: Max moves per game
    """
    # Load model
    if config == "main":
        model = create_model_main()
    else:
        model = create_model_mini()

    print(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Create runner
    runner = ChessRunner(model, device=device)

    # Play games and collect FENs
    all_fens = []
    print(f"Playing {num_games} self-play games...")

    for game_idx in range(num_games):
        fens = play_selfplay_game(runner, max_moves=max_moves)
        all_fens.extend(fens)

        if (game_idx + 1) % 10 == 0:
            print(f"  Completed {game_idx + 1}/{num_games} games ({len(all_fens)} positions)")

    # Write FENs to file
    with open(output_path, "w") as f:
        for fen in all_fens:
            f.write(json.dumps({"fen": fen}) + "\n")

    print(f"\nCollected {len(all_fens)} positions from {num_games} games")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Self-play data collection")
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--out", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument("--config", type=str, choices=["main", "mini"], default="main")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-moves", type=int, default=200, help="Max moves per game")
    args = parser.parse_args()

    run_selfplay(
        model_path=args.ckpt,
        output_path=args.out,
        num_games=args.games,
        config=args.config,
        device=args.device,
        max_moves=args.max_moves,
    )


if __name__ == "__main__":
    main()
