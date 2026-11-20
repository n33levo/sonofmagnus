import argparse
import json
from pathlib import Path

from train.dataset import read_games


def is_tactical(board):
    return any(board.is_capture(move) or board.gives_check(move) for move in board.legal_moves)


def is_complex(board, threshold=40):
    return board.legal_moves.count() >= threshold


def is_endgame(board, max_pieces=10):
    return len(board.piece_map()) <= max_pieces


def extract_hard_positions(pgn_path, output_path, max_games=10000, max_positions=100000, min_elo=2000):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    games, count = 0, 0
    with output.open('w') as stream:
        for game in read_games(pgn_path):
            if games >= max_games or count >= max_positions:
                break
            try:
                elo = min(int(game.headers.get('WhiteElo', 0)), int(game.headers.get('BlackElo', 0)))
            except ValueError:
                continue
            if elo < min_elo or game.errors:
                continue
            games += 1
            board = game.board()
            for move in game.mainline_moves():
                if board.fullmove_number >= 10:
                    tags = {'tactical': is_tactical(board), 'complex': is_complex(board), 'endgame': is_endgame(board)}
                    if any(tags.values()):
                        stream.write(json.dumps({'fen': board.fen(), 'tags': tags}) + '\n')
                        count += 1
                        if count >= max_positions:
                            break
                board.push(move)
    print(f'Curated {count} positions from {games} games')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgn', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-games', type=int, default=10000)
    parser.add_argument('--max-positions', type=int, default=100000)
    parser.add_argument('--min-elo', type=int, default=2000)
    args = parser.parse_args()
    extract_hard_positions(args.pgn, args.out, args.max_games, args.max_positions, args.min_elo)


if __name__ == '__main__':
    main()
