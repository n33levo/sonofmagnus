"""
Rules-correct play loop for chess bot.

Single-pass inference only (no search). Features:
- Legal move masking
- Temperature-annealed sampling
- Timeout guards
- NaN detection
- Fallback strategies
"""

import argparse
import time
from pathlib import Path

import chess
import numpy as np
import torch

from chessio.encode import encode_board
from chessio.mask import sample_legal_move, argmax_legal_move, build_legal_mask, apply_legal_mask
from model import ChessNet, create_model_main, create_model_mini


class ChessRunner:
    """
    Chess bot runner with single-pass inference.
    """

    def __init__(
        self,
        model: ChessNet,
        device: str = "cpu",
        temperature_opening: float = 0.6,
        temperature_midgame: float = 0.4,
        temperature_endgame: float = 0.2,
        move_timeout_ms: float = 1000.0,
    ):
        """
        Args:
            model: Trained ChessNet model
            device: Device to run inference on
            temperature_opening: Sampling temperature for opening (moves 0-10)
            temperature_midgame: Sampling temperature for midgame (moves 11-30)
            temperature_endgame: Sampling temperature for endgame (moves 31+)
            move_timeout_ms: Maximum time per move in milliseconds
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        self.temp_opening = temperature_opening
        self.temp_midgame = temperature_midgame
        self.temp_endgame = temperature_endgame
        self.move_timeout_ms = move_timeout_ms

    def get_temperature(self, move_number: int) -> float:
        """Get temperature based on game phase."""
        if move_number <= 10:
            return self.temp_opening
        elif move_number <= 30:
            return self.temp_midgame
        else:
            return self.temp_endgame

    def select_move(
        self,
        board: chess.Board,
        temperature: float = None,
        deterministic: bool = False,
    ) -> chess.Move:
        """
        Select a move using single-pass inference.

        Args:
            board: Current board state
            temperature: Sampling temperature (None = use auto from move number)
            deterministic: If True, use argmax; if False, sample

        Returns:
            Selected chess.Move
        """
        start_time = time.time()

        # Auto temperature
        if temperature is None:
            temperature = self.get_temperature(board.fullmove_number)

        try:
            # Encode board
            board_tensor = encode_board(board)
            board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)

            # Forward pass
            with torch.no_grad():
                policy_logits, value = self.model(board_tensor)

            # Extract policy logits (B, 8, 8, 73) -> (8, 8, 73)
            policy_logits = policy_logits[0].cpu().numpy()

            # Check for NaN
            if np.isnan(policy_logits).any() or np.isnan(value.cpu().numpy()).any():
                print("WARNING: NaN detected in model output, using fallback")
                return self._fallback_move(board)

            # Select move
            if deterministic:
                move = argmax_legal_move(policy_logits, board)
            else:
                move = sample_legal_move(policy_logits, board, temperature=temperature)

            # Verify move is legal (should always be true)
            if move not in board.legal_moves:
                print(f"WARNING: Selected illegal move {move}, using fallback")
                return self._fallback_move(board)

            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.move_timeout_ms:
                print(f"WARNING: Move selection took {elapsed_ms:.1f}ms (timeout: {self.move_timeout_ms}ms)")

            return move

        except Exception as e:
            print(f"ERROR during move selection: {e}")
            return self._fallback_move(board)

    def _fallback_move(self, board: chess.Board) -> chess.Move:
        """
        Fallback strategy when inference fails.
        Priority: captures > checks > random legal move.
        """
        legal_moves = list(board.legal_moves)

        # Prefer captures
        captures = [m for m in legal_moves if board.is_capture(m)]
        if captures:
            return np.random.choice(captures)

        # Prefer checks
        checks = [m for m in legal_moves if board.gives_check(m)]
        if checks:
            return np.random.choice(checks)

        # Random legal move
        return np.random.choice(legal_moves)

    def play_game(
        self,
        opponent_runner=None,
        max_moves: int = 200,
        opening_fen: str = None,
        verbose: bool = True,
    ) -> dict:
        """
        Play a complete game.

        Args:
            opponent_runner: Another ChessRunner for self-play (None = play against self)
            max_moves: Maximum number of moves before draw
            opening_fen: Starting position (None = standard start)
            verbose: Print game progress

        Returns:
            Dictionary with game info (moves, result, etc.)
        """
        board = chess.Board(opening_fen) if opening_fen else chess.Board()
        move_history = []

        if verbose:
            print(f"Starting game from: {board.fen()}")
            print()

        while not board.is_game_over() and len(move_history) < max_moves:
            # Select runner for current side
            if opponent_runner is None or board.turn == chess.WHITE:
                runner = self
            else:
                runner = opponent_runner

            # Select and make move
            move = runner.select_move(board)
            move_history.append(move)

            if verbose:
                print(f"{len(move_history):3d}. {move.uci():6s} ({board.san(move)})")

            board.push(move)

        # Determine result
        if board.is_checkmate():
            result = "1-0" if board.turn == chess.BLACK else "0-1"
            reason = "checkmate"
        elif board.is_stalemate():
            result = "1/2-1/2"
            reason = "stalemate"
        elif board.is_insufficient_material():
            result = "1/2-1/2"
            reason = "insufficient_material"
        elif board.can_claim_fifty_moves():
            result = "1/2-1/2"
            reason = "fifty_move_rule"
        elif board.can_claim_threefold_repetition():
            result = "1/2-1/2"
            reason = "threefold_repetition"
        elif len(move_history) >= max_moves:
            result = "1/2-1/2"
            reason = "max_moves"
        else:
            result = "1/2-1/2"
            reason = "unknown"

        if verbose:
            print()
            print(f"Game over: {result} ({reason})")
            print(f"Total moves: {len(move_history)}")

        return {
            "result": result,
            "reason": reason,
            "moves": move_history,
            "num_moves": len(move_history),
            "final_fen": board.fen(),
        }


def main():
    parser = argparse.ArgumentParser(description="Chess bot runner")
    parser.add_argument("--ckpt", type=str, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, choices=["main", "mini"], default="main", help="Model config")
    parser.add_argument("--fen", type=str, help="Starting FEN position")
    parser.add_argument("--moves", type=int, default=10, help="Number of moves to play")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of sampling")
    args = parser.parse_args()

    # Create model
    if args.config == "main":
        model = create_model_main()
    else:
        model = create_model_mini()

    # Load checkpoint if provided
    if args.ckpt:
        print(f"Loading checkpoint from {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Create runner
    runner = ChessRunner(model, device=args.device)

    # Play game
    board = chess.Board(args.fen) if args.fen else chess.Board()
    print(f"Starting position: {board.fen()}")
    print()

    for i in range(args.moves):
        if board.is_game_over():
            print("Game over!")
            break

        move = runner.select_move(board, deterministic=args.deterministic)
        print(f"{i+1}. {move.uci()} ({board.san(move)})")
        board.push(move)

    print()
    print(f"Final position: {board.fen()}")


if __name__ == "__main__":
    main()
