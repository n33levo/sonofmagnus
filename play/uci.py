import argparse
import sys

import chess

from play.runner import load_runner


def uci_loop(runner, source=sys.stdin, output=sys.stdout):
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
            move = runner.select_move(board, deterministic=True)
            print(f'bestmove {move.uci() if move else "0000"}', file=output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--backend', default='auto')
    args = parser.parse_args()
    uci_loop(load_runner(args.ckpt, device=args.device, backend=args.backend))


if __name__ == '__main__':
    main()
