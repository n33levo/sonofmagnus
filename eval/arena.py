import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np

from play.runner import load_runner
from model.artifacts import artifact_info


class UciRunner:
    def __init__(self, path='stockfish', elo=None, time_limit=.1):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        options = {key: value for key, value in {'Threads': 1, 'Hash': 256, 'Move Overhead': 10}.items()
                   if key in self.engine.options}
        if elo is not None:
            options.update(UCI_LimitStrength=True, UCI_Elo=elo)
        self.engine.configure(options)
        self.configured_options = options
        self.time_limit = time_limit
        self.limit = chess.engine.Limit(time=time_limit)
        self.last_latency_ms = 0.
        self.game = object()

    def new_game(self):
        self.game = object()

    def set_clock(self, white, black, increment):
        self.limit = chess.engine.Limit(white_clock=white, black_clock=black,
                                       white_inc=increment, black_inc=increment)

    def select_move(self, board, deterministic=True):
        begin = time.perf_counter()
        move = self.engine.play(board, self.limit, game=self.game).move
        self.last_latency_ms = (time.perf_counter() - begin) * 1000
        return move

    def close(self):
        self.engine.quit()


def score_interval(scores, z=1.96):
    n = len(scores)
    p = sum(scores) / n
    # Wilson interval for the mean game score; pair bootstrap is used for match Elo below.
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0., center-half), min(1., center+half)]


def elo_difference(score):
    if score <= 0 or score >= 1:
        return None
    return 400 * math.log10(score / (1-score))


def match_summary(scores):
    n = len(scores)
    if not n:
        raise ValueError('No games played')
    score = float(np.mean(scores))
    pairs = np.asarray(scores).reshape(-1, 2).mean(1)
    rng = np.random.default_rng(0)
    boot = pairs[rng.integers(len(pairs), size=(10000, len(pairs)))].mean(1)
    interval = np.quantile(boot, [.025, .975]).tolist()
    radius = math.sqrt(math.log(40) / (2 * len(pairs)))
    guaranteed = [max(0., score - radius), min(1., score + radius)]
    return {'games': n, 'wins_a': scores.count(1.), 'wins_b': scores.count(0.), 'draws': scores.count(.5),
            'scores_a': scores,
            'score_a': score, 'score_pair_bootstrap_95': interval,
            'score_wilson_95': score_interval(scores), 'elo_difference': elo_difference(score),
            'score_hoeffding_95': guaranteed,
            'elo_difference_hoeffding_95': [elo_difference(p) for p in guaranteed],
            'elo_difference_pair_bootstrap_95': [elo_difference(p) for p in interval]}


def play_match(runner_a, runner_b, num_games=100, time_control=.2, openings=None,
               max_plies=400, pgn_path=None, clock_seconds=None, increment=0., seed=0, names=('A', 'B')):
    if num_games <= 0 or num_games % 2:
        raise ValueError('Use a positive even game count for colour-swapped opening pairs')
    if max_plies <= 0:
        raise ValueError('max_plies must be positive')
    if (clock_seconds is not None and (not math.isfinite(clock_seconds) or clock_seconds <= 0)) or not math.isfinite(increment) or increment < 0:
        raise ValueError('Clock must be positive and increment nonnegative')
    np.random.seed(seed)
    openings = openings or [chess.STARTING_FEN]
    if any(not chess.Board(fen).is_valid() for fen in openings):
        raise ValueError('Invalid opening position')
    scores, latencies, terminations = [], [[], []], Counter()
    forward_start = [getattr(runner, 'forward_calls', None) for runner in (runner_a, runner_b)]
    pgn_file = Path(pgn_path).open('w') if pgn_path else None
    try:
        for game_idx in range(num_games):
            for runner in (runner_a, runner_b):
                if hasattr(runner, 'new_game'):
                    runner.new_game()
            a_white = game_idx % 2 == 0
            board = chess.Board(openings[(game_idx // 2) % len(openings)])
            game = chess.pgn.Game()
            game.setup(board)
            game.headers.update(White=names[0] if a_white else names[1], Black=names[1] if a_white else names[0],
                                Round=str(game_idx + 1))
            if clock_seconds is not None:
                game.headers['TimeControl'] = f'{clock_seconds:g}+{increment:g}'
            node = game
            remaining = [clock_seconds, clock_seconds]
            result, reason = None, None
            for _ in range(max_plies):
                outcome = board.outcome(claim_draw=True)
                if outcome:
                    result, reason = outcome.result(), outcome.termination.name.lower()
                    break
                idx = 0 if board.turn == a_white else 1
                runner = (runner_a, runner_b)[idx]
                if clock_seconds is not None and hasattr(runner, 'set_clock'):
                    runner.set_clock(remaining[0 if a_white else 1], remaining[1 if a_white else 0], increment)
                begin = time.perf_counter()
                move = runner.select_move(board, deterministic=True)
                elapsed = time.perf_counter() - begin
                latencies[idx].append(elapsed * 1000)
                if clock_seconds is not None:
                    remaining[idx] -= elapsed
                    if remaining[idx] <= 0:
                        result = '1/2-1/2' if board.has_insufficient_material(not board.turn) else '0-1' if board.turn else '1-0'
                        reason = 'time_forfeit'
                        break
                    remaining[idx] += increment
                if move not in board.legal_moves:
                    raise ValueError(f'Illegal arena move: {move} at {board.fen()}')
                board.push(move)
                node = node.add_variation(move)
                node.comment = f'{elapsed * 1000:.3f} ms'
            if result is None:
                outcome = board.outcome(claim_draw=True)
                result = outcome.result() if outcome else '1/2-1/2'
                reason = outcome.termination.name.lower() if outcome else 'ply_limit'
            white_score = {'1-0': 1., '0-1': 0., '1/2-1/2': .5}[result]
            scores.append(white_score if a_white else 1-white_score)
            terminations[reason] += 1
            game.headers.update(Result=result, Termination=reason)
            if pgn_file:
                if game_idx:
                    print(file=pgn_file)
                print(game, file=pgn_file)
                pgn_file.flush()
            print(f'Game {game_idx+1}/{num_games}: {result} ({reason})', flush=True)
    finally:
        if pgn_file:
            pgn_file.close()
    summary = match_summary(scores)
    summary['terminations'] = dict(terminations)
    for idx, name in enumerate(('a', 'b')):
        summary[f'latency_{name}_ms'] = {f'p{q}': float(np.percentile(latencies[idx], q)) for q in (50, 95, 99)} if latencies[idx] else {}
        summary[f'moves_{name}'] = len(latencies[idx])
        if forward_start[idx] is not None:
            summary[f'neural_forward_calls_{name}'] = (runner_a, runner_b)[idx].forward_calls - forward_start[idx]
    summary.update(seed=seed, max_plies=max_plies, clock_seconds=clock_seconds, increment=increment,
                   openings_sha256=hashlib.sha256('\n'.join(openings).encode()).hexdigest())
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--config', choices=['main', 'mini'], default='main')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--backend', default='auto')
    parser.add_argument('--stockfish-path', default='stockfish')
    parser.add_argument('--uci-path')
    parser.add_argument('--stockfish-elo', type=int)
    parser.add_argument('--move-time', type=float, default=.1)
    parser.add_argument('--time-control', '--tc', help='Seconds+increment, e.g. 60+0.5')
    parser.add_argument('--openings', help='One FEN per line; each is played with both colours')
    parser.add_argument('--max-plies', type=int, default=400)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='out/arena.json')
    args = parser.parse_args()
    clock, increment = (map(float, args.time_control.split('+')) if args.time_control else (None, 0.))
    a = load_runner(args.a, args.config, args.device, args.backend)
    b = UciRunner(args.uci_path or args.stockfish_path, args.stockfish_elo, args.move_time) if args.b in ('stockfish', 'uci') else load_runner(args.b, args.config, args.device, args.backend)
    for runner in (a, b):
        if hasattr(runner, 'policy'):
            for _ in range(5):
                runner.policy(chess.Board())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    openings = [l.strip() for l in Path(args.openings).read_text().splitlines() if l.strip()] if args.openings else None
    try:
        results = play_match(a, b, args.games, openings=openings, max_plies=args.max_plies,
                             pgn_path=out.with_suffix('.pgn'), clock_seconds=clock, increment=increment, seed=args.seed)
        results.update(a=args.a, b=args.b, stockfish_elo_setting=args.stockfish_elo,
                       stockfish_move_time=args.move_time if clock is None else None,
                       opponent_limit_protocol='clock' if clock is not None else 'fixed_move_time',
                       model_a=artifact_info(args.a), rating_pool='local engine match; not FIDE or Lichess Elo')
        if isinstance(b, UciRunner):
            results['opponent_engine'] = b.engine.id
            results['engine_reset_between_games'] = True
        out.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(results, indent=2))
    finally:
        if isinstance(b, UciRunner):
            b.close()


if __name__ == '__main__':
    main()
