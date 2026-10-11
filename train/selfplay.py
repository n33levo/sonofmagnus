import argparse
import json
from pathlib import Path

import chess

from play.runner import load_runner


def play_selfplay_game(runner, max_moves=400):
    board = chess.Board()
    fens = []
    while not board.is_game_over(claim_draw=True) and len(fens) < max_moves:
        fens.append(board.fen())
        board.push(runner.select_move(board))
    return fens


def run_selfplay(model_path, output_path, num_games=100, config='main', device='auto', max_moves=400, batch_size=64):
    from chessio.mask import sample_legal_move
    if num_games <= 0 or max_moves <= 0 or batch_size <= 0:
        raise ValueError('Games, moves and batch size must be positive')
    runner = load_runner(model_path, config, device)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open('w') as stream:
        for start in range(0, num_games, batch_size):
            boards = [chess.Board() for _ in range(min(batch_size, num_games - start))]
            plies = [0] * len(boards)
            while True:
                active = [idx for idx, board in enumerate(boards)
                          if plies[idx] < max_moves and not board.is_game_over(claim_draw=True)]
                if not active:
                    break
                predictions = runner.batch_policy([boards[idx] for idx in active])
                for idx, (logits, value) in zip(active, predictions):
                    board = boards[idx]
                    move = sample_legal_move(logits, board, runner.get_temperature(board.fullmove_number))
                    stream.write(json.dumps({'fen': board.fen(), 'game_id': f'selfplay-{start + idx}',
                                             'root_fen': board.root().fen(), 'history': [m.uci() for m in board.move_stack],
                                             'move': move.uci(), 'predicted_value': value}) + '\n')
                    board.push(move)
                    plies[idx] += 1
                    count += 1
            print(f'{start + len(boards)} games, {count} positions', flush=True)
    print(f'Self-play: {count} positions -> {output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--config', choices=['main', 'mini'], default='main')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-moves', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    run_selfplay(args.ckpt, args.out, args.games, args.config, args.device, args.max_moves, args.batch_size)


if __name__ == '__main__':
    main()
