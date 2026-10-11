import argparse
import json
import math
import time
from pathlib import Path

import chess
import numpy as np

from model.artifacts import artifact_info
from play.runner import load_runner


def play_selfplay_game(runner, max_moves=400):
    board = chess.Board()
    fens = []
    while not board.is_game_over(claim_draw=True) and len(fens) < max_moves:
        fens.append(board.fen())
        board.push(runner.select_move(board))
    return fens


def run_selfplay(model_path, output_path, num_games=100, config='main', device='auto', max_moves=400, batch_size=64,
                 positions=None, seed=0):
    from chessio.mask import sample_legal_move, get_legal_move_probs
    if num_games <= 0 or max_moves <= 0 or batch_size <= 0 or (positions is not None and positions <= 0):
        raise ValueError('Games, moves and batch size must be positive')
    np.random.seed(seed)
    begin = time.perf_counter()
    frozen = artifact_info(model_path)
    runner = load_runner(model_path, config, device)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_suffix(output.suffix + '.part')
    count, started, finished, neural_positions = 0, 0, 0, 0
    with temporary.open('x') as stream:
        for start in range(0, num_games, batch_size):
            boards = [chess.Board() for _ in range(min(batch_size, num_games - start))]
            plies = [0] * len(boards)
            while True:
                active = [idx for idx, board in enumerate(boards)
                          if plies[idx] < max_moves and not board.is_game_over(claim_draw=True)]
                if not active:
                    break
                if positions is not None:
                    active = active[:positions - count]
                    if not active:
                        break
                predictions = runner.batch_policy([boards[idx] for idx in active])
                if len(predictions) != len(active):
                    raise ValueError('Missing self-play predictions')
                neural_positions += len(active)
                for idx, (logits, value) in zip(active, predictions):
                    board = boards[idx]
                    move = sample_legal_move(logits, board, runner.get_temperature(board.fullmove_number))
                    probabilities = get_legal_move_probs(logits, board)
                    entropy = (-sum(p * math.log(max(p, 1e-300)) for p in probabilities.values())
                               / math.log(len(probabilities))) if len(probabilities) > 1 else 0.
                    stream.write(json.dumps({'fen': board.fen(), 'game_id': f'selfplay-{start + idx}',
                                             'root_fen': board.root().fen(), 'history': [m.uci() for m in board.move_stack],
                                             'move': move.uci(), 'predicted_value': value,
                                             'policy_entropy': max(0., min(1., entropy))}) + '\n')
                    board.push(move)
                    plies[idx] += 1
                    count += 1
            started += sum(ply > 0 for ply in plies)
            finished += sum(board.is_game_over(claim_draw=True) for board in boards)
            print(f'{started} games, {count} positions', flush=True)
            if positions is not None and count == positions:
                break
    if (positions is not None and count != positions) or artifact_info(model_path) != frozen:
        raise ValueError('Incomplete self-play corpus or changed model')
    temporary.replace(output)
    output.with_suffix('.json').write_text(json.dumps({'positions': count, 'games_started': started,
        'games_finished': finished, 'unfinished_games': started - finished, 'seed': seed,
        'requested_games': num_games, 'requested_positions': positions, 'max_plies': max_moves, 'batch_size': batch_size,
        'seconds': time.perf_counter() - begin,
        'model': frozen, 'artifact': artifact_info(output), 'neural_input_positions': neural_positions,
        'neural_forward_calls': getattr(runner, 'forward_calls', None),
        'values': 'Predicted values are recorded separately; unfinished games have no fabricated outcome target',
        'policy_entropy': 'Legal policy entropy at temperature 1, divided by log of the legal-move count',
        'runtime_source': {path: artifact_info(path)['sha256'] for path in
            ('train/selfplay.py', 'play/runner.py', 'play/mlx.py', 'chessio/lc0.py', 'chessio/mask.py', 'chessio/policy_map.py')}}, indent=2) + '\n')
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
    parser.add_argument('--positions', type=int)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run_selfplay(args.ckpt, args.out, args.games, args.config, args.device, args.max_moves, args.batch_size,
                 args.positions, args.seed)


if __name__ == '__main__':
    main()
