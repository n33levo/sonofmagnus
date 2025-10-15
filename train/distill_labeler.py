"""
Engine distillation labeler using Stockfish.

IMPORTANT: This module uses chess.engine (Stockfish) but is ONLY for
offline training data generation. It is NEVER called during play/inference.

Generates:
- Top-k move distributions from engine analysis
- Position values (centipawns -> normalized)
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import chess
import chess.engine
from tqdm import tqdm


def centipawns_to_value(cp: int) -> float:
    """
    Convert centipawns to value in [-1, 1].

    Uses tanh scaling: value = tanh(cp / 400)
    """
    import math
    return math.tanh(cp / 400.0)


def label_position(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int = 14,
    time_limit: float = 0.1,
    top_k: int = 5,
) -> dict:
    """
    Generate engine labels for a position.

    Args:
        board: Position to analyze
        engine: Stockfish engine
        depth: Search depth
        time_limit: Time limit in seconds
        top_k: Number of top moves to include

    Returns:
        Dictionary with:
            - fen: Position FEN
            - move_probs: Top-k moves with normalized scores
            - value: Position evaluation in [-1, 1]
    """
    # Run multipv analysis
    info = engine.analyse(
        board,
        chess.engine.Limit(depth=depth, time=time_limit),
        multipv=top_k,
    )

    # Extract move scores
    move_scores = []
    for pv_info in info:
        move = pv_info["pv"][0]
        score = pv_info["score"].relative

        # Convert score to centipawns
        if score.is_mate():
            # Mate scores: very high value
            cp = 10000 if score.mate() > 0 else -10000
        else:
            cp = score.score()

        move_scores.append((move, cp))

    # Normalize to probabilities using softmax over centipawns
    import math
    import numpy as np

    cps = np.array([cp for _, cp in move_scores], dtype=np.float32)
    # Temperature scaling
    cps = cps / 100.0  # Scale down
    cps = cps - np.max(cps)  # Numerical stability
    exp_cps = np.exp(cps)
    probs = exp_cps / np.sum(exp_cps)

    move_probs = {
        move.uci(): float(prob)
        for (move, _), prob in zip(move_scores, probs)
    }

    # Position value (from best move score)
    best_cp = move_scores[0][1]
    value = centipawns_to_value(best_cp)

    return {
        "fen": board.fen(),
        "move_probs": move_probs,
        "value": value,
    }


def label_positions_from_file(
    input_path: str,
    output_path: str,
    engine_path: str = "stockfish",
    depth: int = 14,
    time_limit: float = 0.1,
    top_k: int = 5,
    max_positions: Optional[int] = None,
):
    """
    Label positions from input JSONL file.

    Args:
        input_path: Input JSONL with FENs
        output_path: Output JSONL with labels
        engine_path: Path to Stockfish binary
        depth: Search depth
        time_limit: Time limit per position
        top_k: Number of top moves
        max_positions: Maximum positions to label
    """
    # Initialize engine
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    print(f"Loaded engine: {engine_path}")

    # Process positions
    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in tqdm(fin, desc="Labeling positions"):
            if max_positions and count >= max_positions:
                break

            item = json.loads(line)
            fen = item.get("fen", item.get("position"))

            if not fen:
                continue

            board = chess.Board(fen)

            # Skip terminal positions
            if board.is_game_over():
                continue

            try:
                label = label_position(
                    board=board,
                    engine=engine,
                    depth=depth,
                    time_limit=time_limit,
                    top_k=top_k,
                )

                fout.write(json.dumps(label) + "\n")
                count += 1

                if count % 100 == 0:
                    fout.flush()

            except Exception as e:
                print(f"Error labeling {fen}: {e}")
                continue

    engine.quit()
    print(f"\nLabeled {count} positions -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate engine labels for training (OFFLINE ONLY)"
    )
    parser.add_argument("--fens", type=str, required=True, help="Input JSONL with FENs")
    parser.add_argument("--out", type=str, required=True, help="Output JSONL with labels")
    parser.add_argument("--engine", type=str, default="stockfish", help="Stockfish binary path")
    parser.add_argument("--depth", type=int, default=14, help="Search depth")
    parser.add_argument("--time", type=float, default=0.1, help="Time limit per position")
    parser.add_argument("--topk", type=int, default=5, help="Number of top moves")
    parser.add_argument("--max", type=int, help="Max positions to label")
    parser.add_argument("--preset", type=str, choices=["fast", "deep"], help="Preset config")
    args = parser.parse_args()

    # Apply presets
    if args.preset == "fast":
        args.depth = 12
        args.time = 0.05
        args.topk = 5
    elif args.preset == "deep":
        args.depth = 18
        args.time = 0.5
        args.topk = 12

    print(f"Labeling config:")
    print(f"  Depth: {args.depth}")
    print(f"  Time: {args.time}s")
    print(f"  Top-K: {args.topk}")

    label_positions_from_file(
        input_path=args.fens,
        output_path=args.out,
        engine_path=args.engine,
        depth=args.depth,
        time_limit=args.time,
        top_k=args.topk,
        max_positions=args.max,
    )


if __name__ == "__main__":
    main()
