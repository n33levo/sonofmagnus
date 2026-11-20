import argparse
from contextlib import nullcontext
import json
import os
import sys
import time
from pathlib import Path

import chess

from play.runner import load_runner


def uci_loop(runner, source=sys.stdin, output=sys.stdout, audit=None):
    board = chess.Board()
    for line in source:
        words = line.split()
        if not words:
            continue
        command = words[0]
        if command == 'quit':
            break
        if command == 'uci':
            print('id name SonOfMagnus\nid author n33levo\nuciok', file=output, flush=True)
        elif command == 'isready':
            print('readyok', file=output, flush=True)
        elif command == 'ucinewgame':
            board.reset()
        elif command == 'position':
            split = words.index('moves') if 'moves' in words else len(words)
            board = chess.Board() if words[1] == 'startpos' else chess.Board(' '.join(words[2:split]))
            for move in words[split+1:]:
                board.push_uci(move)
        elif command == 'go':
            calls = runner.forward_calls if audit is not None else 0
            move = runner.select_move(board, deterministic=True)
            if (move is None and not board.is_game_over()) or (move is not None and move not in board.legal_moves):
                raise ValueError('Engine returned an illegal move')
            if audit is not None:
                forwards = runner.forward_calls - calls
                if forwards != int(move is not None):
                    raise ValueError('Expected one network pass for each move')
                audit.write(json.dumps({'utc_seconds': time.time(), 'pid': os.getpid(), 'fen': board.fen(),
                    'root_fen': board.root().fen(), 'history': [m.uci() for m in board.move_stack],
                    'move': move.uci() if move else None, 'neural_forward_calls': forwards,
                    'latency_ms': runner.last_latency_ms}, separators=(',', ':')) + '\n')
                audit.flush()
            print(f'bestmove {move.uci() if move else "0000"}', file=output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--backend', default='auto')
    parser.add_argument('--audit-log', type=Path)
    args = parser.parse_args()
    if args.audit_log:
        args.audit_log.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_log.open('a') if args.audit_log else nullcontext() as audit:
        uci_loop(load_runner(args.ckpt, device=args.device, backend=args.backend), audit=audit)


if __name__ == '__main__':
    main()
