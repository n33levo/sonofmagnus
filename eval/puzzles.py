import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import chess

from play.runner import load_runner
from eval.statistics import binomial_interval
from model.artifacts import artifact_info


def puzzle_moves(puzzle):
    moves = puzzle.get('moves', puzzle.get('Moves'))
    return moves.split() if isinstance(moves, str) else moves


def solve_puzzle(runner, fen, solution_moves, max_attempts=1, first_move_only=False):
    if max_attempts != 1:
        raise ValueError('Puzzle evaluation uses exactly one deterministic attempt')
    board = chess.Board(fen)
    if len(solution_moves) < 2 or len(solution_moves) % 2:
        raise ValueError('Lichess puzzles need a setup move and at least one response')
    for idx, uci in enumerate(solution_moves):
        expected = chess.Move.from_uci(uci)
        if expected not in board.legal_moves:
            raise ValueError(f'Illegal puzzle solution: {uci} at {board.fen()}')
        # Lichess stores the opponent setup first, followed by the solver response.
        if idx % 2 == 1:
            predicted = runner.select_move(board, deterministic=True)
            if predicted != expected:
                if predicted not in board.legal_moves:
                    return False
                alternative = board.copy()
                alternative.push(predicted)
                return alternative.is_checkmate()
            if first_move_only:
                return True
        board.push(expected)
    return True


def read_puzzles(path):
    path = Path(path)
    raw = path.open('rb')
    if path.suffix == '.zst':
        import zstandard
        stream = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding='utf-8')
    else:
        stream = io.TextIOWrapper(raw, encoding='utf-8')
    with raw, stream:
        if '.csv' in path.suffixes:
            yield from csv.DictReader(stream)
        else:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def evaluate_runner(runner, puzzles_path, max_puzzles=None, first_move_only=False):
    buckets = defaultdict(lambda: {'correct': 0, 'total': 0})
    bands = defaultdict(lambda: {'correct': 0, 'total': 0})
    total = 0
    for puzzle in read_puzzles(puzzles_path):
        if max_puzzles is not None and total >= max_puzzles:
            break
        rating = int(puzzle.get('rating', puzzle.get('Rating', 1500)))
        key = (rating // 100) * 100
        solved = solve_puzzle(runner, puzzle.get('fen', puzzle.get('FEN')),
                              puzzle_moves(puzzle), first_move_only=first_move_only)
        buckets[key]['correct'] += solved
        buckets[key]['total'] += 1
        if rating >= 1500:
            band = ('1500-1800', '1800-2100', '2100-2400', '2400-2700', '2700+')[min((rating - 1500) // 300, 4)]
            bands[band]['correct'] += solved
            bands[band]['total'] += 1
        total += 1
    if not total:
        raise ValueError('Empty puzzle dataset')
    results = {key: {**stats, 'accuracy': stats['correct'] / stats['total']} for key, stats in buckets.items()}
    correct = sum(s['correct'] for s in buckets.values())
    results['overall'] = {'correct': correct, 'total': total, 'accuracy': correct / total}
    results['protocol'] = 'first_response' if first_move_only else 'complete_line'
    results['bands'] = {band: {**stats, 'accuracy': stats['correct'] / stats['total'],
        'clopper_pearson_simultaneous_95': binomial_interval(stats['correct'], stats['total'], .05 / 5)}
        for band, stats in bands.items()}
    return results


def evaluate_puzzles(model_path, puzzles_path, config='main', device='auto', max_puzzles=None):
    return evaluate_runner(load_runner(model_path, config, device), puzzles_path, max_puzzles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--puzzles', required=True)
    parser.add_argument('--config', choices=['main', 'mini'], default='main')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max', type=int)
    parser.add_argument('--first-move-only', action='store_true')
    parser.add_argument('--out')
    args = parser.parse_args()
    model, dataset = artifact_info(args.ckpt), artifact_info(args.puzzles)
    runner = load_runner(args.ckpt, args.config, args.device)
    results = evaluate_runner(runner, args.puzzles, args.max, args.first_move_only)
    if artifact_info(args.ckpt) != model or artifact_info(args.puzzles) != dataset:
        raise ValueError('Model or puzzle dataset changed during evaluation')
    results.update(model=model, dataset=dataset,
                   neural_forward_calls=getattr(runner, 'forward_calls', None),
                   runtime_source={path: artifact_info(path)['sha256'] for path in
                       ('eval/puzzles.py', 'eval/statistics.py', 'play/runner.py', 'play/mlx.py', 'chessio/lc0.py')})
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
