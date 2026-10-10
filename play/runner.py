import argparse
import time

import chess
import torch

from chessio.encode import encode_board_batch
from chessio.mask import sample_legal_move, get_legal_move_probs
from model import load_model, resolve_device


def load_runner(path, config='main', device='auto', backend='auto'):
    if str(path).endswith('.onnx'):
        from play.onnx import OnnxRunner
        return OnnxRunner(path)
    if str(path).endswith('.pb.gz'):
        from play.pretrained import PretrainedRunner
        if backend == 'auto' and device == 'cpu':
            backend = 'blas'
        return PretrainedRunner(path, backend)
    return ChessRunner(load_model(path, config), device)


class ChessRunner:
    def __init__(self, model, device='auto', temperature_opening=.6,
                 temperature_midgame=.4, temperature_endgame=.2, move_timeout_ms=1000.):
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.temp_opening, self.temp_midgame, self.temp_endgame = (
            temperature_opening, temperature_midgame, temperature_endgame)
        self.move_timeout_ms = move_timeout_ms
        self.last_latency_ms = 0.
        self.forward_calls = 0

    def get_temperature(self, move_number):
        return self.temp_opening if move_number <= 10 else self.temp_midgame if move_number <= 30 else self.temp_endgame

    @torch.inference_mode()
    def batch_policy(self, boards):
        tensor = torch.from_numpy(encode_board_batch(boards)).to(self.device)
        tensor = tensor.to(next(self.model.parameters()).dtype)
        logits, value = self.model(tensor)
        self.forward_calls += 1
        if not torch.isfinite(value).all():
            raise ValueError('Non-finite value output')
        logits, value = logits.float().cpu().numpy(), value.float().cpu().numpy().reshape(-1)
        return list(zip(logits, value.tolist()))

    def policy(self, board):
        return self.batch_policy([board])[0]

    def move_probs(self, board):
        logits, _ = self.policy(board)
        return get_legal_move_probs(logits, board)

    def select_move(self, board, temperature=None, deterministic=False):
        if board.is_game_over(claim_draw=True):
            return None
        start = time.perf_counter()
        logits, _ = self.policy(board)
        temp = 0 if deterministic else self.get_temperature(board.fullmove_number) if temperature is None else temperature
        move = sample_legal_move(logits, board, temp)
        self.last_latency_ms = (time.perf_counter() - start) * 1000
        return move

    def play_game(self, opponent_runner=None, max_moves=200, opening_fen=None, verbose=True):
        board = chess.Board(opening_fen) if opening_fen else chess.Board()
        moves = []
        while not board.is_game_over(claim_draw=True) and len(moves) < max_moves:
            runner = self if board.turn or opponent_runner is None else opponent_runner
            move = runner.select_move(board)
            if move not in board.legal_moves:
                raise ValueError(f'Illegal move: {move}')
            if verbose:
                print(f'{len(moves) + 1}. {board.san(move)}')
            moves.append(move)
            board.push(move)
        outcome = board.outcome(claim_draw=True)
        return {'result': outcome.result() if outcome else '1/2-1/2',
                'reason': outcome.termination.name.lower() if outcome else 'max_moves',
                'moves': moves, 'num_moves': len(moves), 'final_fen': board.fen()}


def play_interactive(runner, board, input_fn=input):
    while not board.is_game_over(claim_draw=True):
        print(board)
        if board.turn == chess.WHITE:
            text = input_fn('Your move (SAN or UCI, quit to exit): ').strip()
            if text == 'quit':
                return
            try:
                move = board.parse_san(text)
            except ValueError:
                print('Illegal move')
                continue
        else:
            move = runner.select_move(board, deterministic=True)
            print(f'Engine: {board.san(move)}')
        board.push(move)
    print(board.result(claim_draw=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--config', choices=['main', 'mini'], default='main')
    parser.add_argument('--fen')
    parser.add_argument('--moves', type=int, help='Play this many plies automatically; otherwise play interactively')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    runner = load_runner(args.ckpt, args.config, args.device)
    board = chess.Board(args.fen) if args.fen else chess.Board()
    if args.moves is None:
        play_interactive(runner, board)
        return
    for _ in range(args.moves):
        if board.is_game_over(claim_draw=True):
            break
        move = runner.select_move(board, deterministic=args.deterministic)
        print(f'{move.uci()} ({board.san(move)})')
        board.push(move)
    print(board.fen())


if __name__ == '__main__':
    main()
