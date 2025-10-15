"""
Curate "hard" positions for distillation.

Hard positions include:
- Tactical positions (captures, checks, pins)
- Complex middlegames (high branching factor)
- Endgames with few pieces
"""

import argparse
import io
import json
import random

import chess
import chess.pgn
from tqdm import tqdm


def is_tactical(board: chess.Board) -> bool:
    """Check if position has tactical elements."""
    # Has captures or checks available
    has_captures = any(board.is_capture(m) for m in board.legal_moves)
    has_checks = any(board.gives_check(m) for m in board.legal_moves)
    return has_captures or has_checks


def is_complex(board: chess.Board, threshold: int = 40) -> bool:
    """Check if position has high branching factor."""
    return board.legal_moves.count() >= threshold


def is_endgame(board: chess.Board, max_pieces: int = 10) -> bool:
    """Check if position is an endgame."""
    num_pieces = len(board.piece_map())
    return num_pieces <= max_pieces


def extract_hard_positions(
    pgn_path: str,
    output_path: str,
    max_games: int = 10000,
    max_positions: int = 100000,
    min_elo: int = 2000,
):
    """
    Extract hard positions from PGN file.

    Args:
        pgn_path: Input PGN file
        output_path: Output JSONL file
        max_games: Max games to process
        max_positions: Max positions to extract
        min_elo: Minimum Elo for games
    """
    positions = []
    games_processed = 0

    print(f"Extracting hard positions from {pgn_path}...")

    def game_iterator():
        if pgn_path.endswith(".zst"):
            import zstandard as zstd

            with open(pgn_path, "rb") as fh:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(fh) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
                    while True:
                        game = chess.pgn.read_game(text_stream)
                        if game is None:
                            break
                        yield game
        else:
            with open(pgn_path, encoding="utf-8", errors="ignore") as fh:
                while True:
                    game = chess.pgn.read_game(fh)
                    if game is None:
                        break
                    yield game

    for game in game_iterator():
        if games_processed >= max_games or len(positions) >= max_positions:
            break

        # Check Elo
        white_elo = int(game.headers.get("WhiteElo", 0))
        black_elo = int(game.headers.get("BlackElo", 0))
        avg_elo = (white_elo + black_elo) / 2

        if avg_elo < min_elo:
            continue

        games_processed += 1

        # Traverse game
        board = game.board()
        for move in game.mainline_moves():
            # Skip early opening
            if board.fullmove_number < 10:
                board.push(move)
                continue

            # Check if position is "hard"
            if is_tactical(board) or is_complex(board) or is_endgame(board):
                positions.append({
                    "fen": board.fen(),
                    "tags": {
                        "tactical": is_tactical(board),
                        "complex": is_complex(board),
                        "endgame": is_endgame(board),
                    }
                })

            board.push(move)

        if games_processed % 100 == 0:
            print(f"  Processed {games_processed} games, found {len(positions)} hard positions")

    # Shuffle and limit
    random.shuffle(positions)
    positions = positions[:max_positions]

    # Write to file
    with open(output_path, "w") as f:
        for pos in positions:
            f.write(json.dumps(pos) + "\n")

    print(f"\nWrote {len(positions)} hard positions to {output_path}")

    # Statistics
    tactical_count = sum(1 for p in positions if p["tags"]["tactical"])
    complex_count = sum(1 for p in positions if p["tags"]["complex"])
    endgame_count = sum(1 for p in positions if p["tags"]["endgame"])

    print(f"  Tactical: {tactical_count}")
    print(f"  Complex: {complex_count}")
    print(f"  Endgame: {endgame_count}")


def main():
    parser = argparse.ArgumentParser(description="Curate hard positions for distillation")
    parser.add_argument("--pgn", type=str, required=True, help="Input PGN file")
    parser.add_argument("--out", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--max-games", type=int, default=10000, help="Max games to process")
    parser.add_argument("--max-positions", type=int, default=100000, help="Max positions")
    parser.add_argument("--min-elo", type=int, default=2000, help="Minimum Elo")
    args = parser.parse_args()

    extract_hard_positions(
        pgn_path=args.pgn,
        output_path=args.out,
        max_games=args.max_games,
        max_positions=args.max_positions,
        min_elo=args.min_elo,
    )


if __name__ == "__main__":
    main()
