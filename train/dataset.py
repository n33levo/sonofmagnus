"""
Dataset loaders for chess training.

Supports:
- Lichess CC0 PGN parsing
- Position sampling (midgame, tactical, etc.)
- Behavioral cloning and distillation labels
"""

import io
import json
import random
from pathlib import Path
from typing import Iterator, Optional

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset, IterableDataset

from chessio.encode import encode_board
from chessio.policy_map import build_policy_target, build_policy_distribution, build_q_value_target


def mirror_move(move: chess.Move) -> chess.Move:
    """Mirror a move across the horizontal axis (swap colors perspective)."""
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
        drop=move.drop,
    )


class ChessPositionDataset(Dataset):
    """
    Dataset of chess positions with move/value labels.
    Loads from JSONL format.
    """

    def __init__(
        self,
        data_path: str,
        augment: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_path: Path to JSONL file with positions
            augment: Apply random augmentations (color swap)
            max_samples: Maximum number of samples to load (None = all)
        """
        self.data_path = Path(data_path)
        self.augment = augment
        self.positions = []

        # Load data
        with open(self.data_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.positions.append(json.loads(line))

        print(f"Loaded {len(self.positions)} positions from {data_path}")

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            Dictionary with:
                - board_tensor: (18, 8, 8) float32
                - policy_target: (8, 8, 73) float32 (one-hot or distribution)
                - value_target: float32 scalar
        """
        item = self.positions[idx]

        # Parse position
        base_board = chess.Board(item["fen"])
        augment = self.augment and random.random() < 0.5

        board = base_board.mirror() if augment else base_board

        # Encode board
        board_tensor = encode_board(board)

        # Build policy target
        if "move" in item:
            # Single move (one-hot)
            move = chess.Move.from_uci(item["move"])
            if augment:
                move = mirror_move(move)
            policy_target = build_policy_target(move, board)
        elif "move_probs" in item:
            # Distribution (from distillation)
            move_probs = {}
            for uci, prob in item["move_probs"].items():
                move_obj = chess.Move.from_uci(uci)
                if augment:
                    move_obj = mirror_move(move_obj)
                move_probs[move_obj] = prob
            policy_target = build_policy_distribution(move_probs, board)
        else:
            raise ValueError(f"Position {idx} has no move or move_probs")

        # Value target
        value_target = float(item.get("value", 0.0))
        if augment:
            value_target = -value_target

        result = {
            "board_tensor": torch.from_numpy(board_tensor),
            "policy_target": torch.from_numpy(policy_target),
            "value_target": torch.tensor(value_target, dtype=torch.float32),
        }

        # Q-value target (optional)
        if "q_values" in item:
            q_value_dict = {}
            for uci, q_val in item["q_values"].items():
                move_obj = chess.Move.from_uci(uci)
                if augment:
                    move_obj = mirror_move(move_obj)
                    # Mirror Q-value: Q'(m) = 1 - Q(m)
                    q_val = 1.0 - q_val
                q_value_dict[move_obj] = q_val

            q_value_target = build_q_value_target(q_value_dict, board)
            result["q_value_target"] = torch.from_numpy(q_value_target)

        return result


class PGNStreamDataset(IterableDataset):
    """
    Iterable dataset that streams positions from PGN files.
    Useful for large Lichess databases.
    """

    def __init__(
        self,
        pgn_path: str,
        min_elo: int = 1800,
        sample_rate: float = 1.0,
        midgame_only: bool = True,
        max_positions: Optional[int] = None,
    ):
        """
        Args:
            pgn_path: Path to PGN file (can be .pgn or .pgn.zst)
            min_elo: Minimum Elo to include games
            sample_rate: Probability of including each position
            midgame_only: Only sample positions from moves 10-40
            max_positions: Stop after this many positions
        """
        self.pgn_path = Path(pgn_path)
        self.min_elo = min_elo
        self.sample_rate = sample_rate
        self.midgame_only = midgame_only
        self.max_positions = max_positions

    def _parse_pgn(self) -> Iterator[dict]:
        """Parse PGN file and yield positions."""
        count = 0

        # Handle compressed PGN
        if self.pgn_path.suffix == ".zst":
            import zstandard as zstd
            with open(self.pgn_path, "rb") as fh:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(fh) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                    yield from self._parse_pgn_stream(text_stream, count)
        else:
            with open(self.pgn_path) as f:
                yield from self._parse_pgn_stream(f, count)

    def _parse_pgn_stream(self, stream, count) -> Iterator[dict]:
        """Parse PGN from a text stream."""
        while True:
            if self.max_positions and count >= self.max_positions:
                break

            game = chess.pgn.read_game(stream)
            if game is None:
                break

            # Check Elo requirement
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
            avg_elo = (white_elo + black_elo) / 2
            if avg_elo < self.min_elo:
                continue

            # Extract result value
            result = game.headers.get("Result", "*")
            if result == "1-0":
                game_value = 1.0
            elif result == "0-1":
                game_value = -1.0
            else:
                game_value = 0.0

            # Traverse game
            board = game.board()
            move_num = 0

            for move in game.mainline_moves():
                move_num += 1

                # Midgame filter
                if self.midgame_only and (move_num < 10 or move_num > 40):
                    board.push(move)
                    continue

                # Sampling
                if random.random() > self.sample_rate:
                    board.push(move)
                    continue

                # Yield position
                value = game_value if board.turn == chess.WHITE else -game_value

                yield {
                    "fen": board.fen(),
                    "move": move.uci(),
                    "value": value,
                }

                count += 1
                if self.max_positions and count >= self.max_positions:
                    break

                board.push(move)

    def __iter__(self):
        """Iterate over positions."""
        for item in self._parse_pgn():
            board = chess.Board(item["fen"])
            board_tensor = encode_board(board)

            move = chess.Move.from_uci(item["move"])
            policy_target = build_policy_target(move, board)

            value_target = item["value"]

            yield {
                "board_tensor": torch.from_numpy(board_tensor),
                "policy_target": torch.from_numpy(policy_target),
                "value_target": torch.tensor(value_target, dtype=torch.float32),
            }


def build_jsonl_from_pgn(
    pgn_path: str,
    output_path: str,
    min_elo: int = 1800,
    sample_rate: float = 0.1,
    max_positions: int = 1_000_000,
):
    """
    Convert PGN to JSONL format for faster loading.

    Args:
        pgn_path: Input PGN file
        output_path: Output JSONL file
        min_elo: Minimum Elo
        sample_rate: Sampling rate
        max_positions: Maximum positions
    """
    dataset = PGNStreamDataset(
        pgn_path=pgn_path,
        min_elo=min_elo,
        sample_rate=sample_rate,
        max_positions=max_positions,
    )

    count = 0
    with open(output_path, "w") as f:
        for item in dataset._parse_pgn():
            f.write(json.dumps(item) + "\n")
            count += 1

            if count % 10000 == 0:
                print(f"Processed {count} positions...")

    print(f"Wrote {count} positions to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build dataset from PGN")
    parser.add_argument("--pgn", type=str, required=True, help="Input PGN file")
    parser.add_argument("--out", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--elo-min", type=int, default=1800, help="Minimum Elo")
    parser.add_argument("--sample-rate", type=float, default=0.1, help="Sampling rate")
    parser.add_argument("--max-positions", type=int, default=1_000_000, help="Max positions")
    args = parser.parse_args()

    build_jsonl_from_pgn(
        pgn_path=args.pgn,
        output_path=args.out,
        min_elo=args.elo_min,
        sample_rate=args.sample_rate,
        max_positions=args.max_positions,
    )
