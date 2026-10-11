import argparse
from array import array
import hashlib
import io
import json
import random
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from chessio.encode import encode_board
from chessio.mask import build_legal_mask
from chessio.policy_map import build_policy_target, build_policy_distribution, build_q_value_target


def mirror_move(move):
    return chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square), promotion=move.promotion, drop=move.drop)


def position_sample(item, augment=False):
    board = chess.Board(item['fen'])
    if augment:
        board = board.mirror()

    def moves(values):
        return {(mirror_move(chess.Move.from_uci(m)) if augment else chess.Move.from_uci(m)): v
                for m, v in values.items()}

    if 'move_probs' in item:
        policy = build_policy_distribution(moves(item['move_probs']), board)
    else:
        move = chess.Move.from_uci(item['move'])
        policy = build_policy_target(mirror_move(move) if augment else move, board)
    value = float(item.get('value', 0))
    if not np.isfinite(value) or not -1 <= value <= 1:
        raise ValueError('Value target must be in [-1, 1] from the side to move')
    weight = float(item.get('weight', 1))
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError('Sample weight must be finite and positive')
    # Mirroring swaps both colours and the turn, so relative targets do not change.
    return {
        'board_tensor': torch.from_numpy(encode_board(board)),
        'policy_target': torch.from_numpy(policy),
        'legal_mask': torch.from_numpy(build_legal_mask(board)),
        'value_target': torch.tensor(value, dtype=torch.float32),
        'value_mask': torch.tensor('value' in item),
        'weight': torch.tensor(weight, dtype=torch.float32),
        'q_value_target': torch.from_numpy(build_q_value_target(moves(item.get('q_values', {})), board)),
    }


class ChessPositionDataset(Dataset):
    def __init__(self, data_path, augment=True, max_samples=None):
        self.data_path = Path(data_path)
        self.augment = augment
        self.weight = 1.
        self.offsets = array('Q')
        with self.data_path.open('rb') as stream:
            while max_samples is None or len(self.offsets) < max_samples:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        self.stream = None

    def __len__(self):
        return len(self.offsets)

    def __getstate__(self):
        return {**self.__dict__, 'stream': None}

    def read_row(self, idx):
        if self.stream is None:
            self.stream = self.data_path.open('rb')
        self.stream.seek(self.offsets[idx])
        item = json.loads(self.stream.readline())
        item['weight'] = item.get('weight', 1.) * self.weight
        return item

    def __getitem__(self, idx):
        item = self.read_row(idx)
        return position_sample(item, self.augment and random.random() < 0.5)


def read_games(pgn_path):
    path = Path(pgn_path)
    if path.suffix == '.zst':
        import zstandard
        with path.open('rb') as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
            with io.TextIOWrapper(reader, encoding='utf-8') as stream:
                while (game := chess.pgn.read_game(stream)) is not None:
                    yield game
    else:
        with path.open(encoding='utf-8-sig') as stream:
            while (game := chess.pgn.read_game(stream)) is not None:
                yield game


class PGNStreamDataset(IterableDataset):
    def __init__(self, pgn_path, min_elo=1800, sample_rate=1.0,
                 midgame_only=True, max_positions=None, seed=0, include_history=False):
        if not 0 <= sample_rate <= 1:
            raise ValueError('Sample rate must be in [0, 1]')
        self.pgn_path, self.min_elo, self.sample_rate = pgn_path, min_elo, sample_rate
        self.midgame_only, self.max_positions, self.seed = midgame_only, max_positions, seed
        self.include_history = include_history

    def _parse_pgn(self):
        rng = random.Random(self.seed)
        worker = get_worker_info()
        count = 0
        for game_idx, game in enumerate(read_games(self.pgn_path)):
            if worker and game_idx % worker.num_workers != worker.id:
                continue
            if self.max_positions is not None and count >= self.max_positions:
                break
            try:
                elo = min(int(game.headers.get('WhiteElo', 0)), int(game.headers.get('BlackElo', 0)))
            except ValueError:
                continue
            if elo < self.min_elo or game.errors or game.headers.get('Result') not in ('1-0', '0-1', '1/2-1/2'):
                continue
            game_value = {'1-0': 1., '0-1': -1., '1/2-1/2': 0.}[game.headers['Result']]
            line = list(game.mainline_moves())
            game_id = hashlib.blake2b((game.board().fen() + ' '.join(m.uci() for m in line)).encode(), digest_size=8).hexdigest()
            board = game.board()
            for move in line:
                if (not self.midgame_only or 10 <= board.fullmove_number <= 40) and rng.random() < self.sample_rate:
                    item = {'fen': board.fen(), 'move': move.uci(),
                           'value': game_value if board.turn else -game_value, 'game_id': game_id,
                           'weight': 3. if elo >= 2600 else 2. if elo >= 2400 else 1.5 if elo >= 2200 else 1.}
                    if self.include_history:
                        item.update(root_fen=board.root().fen(), history=[m.uci() for m in board.move_stack])
                    yield item
                    count += 1
                    if self.max_positions is not None and count >= self.max_positions:
                        break
                board.push(move)

    def __iter__(self):
        for item in self._parse_pgn():
            yield position_sample(item)


def build_jsonl_from_pgn(pgn_path, output_path, min_elo=1800, sample_rate=.1,
                         max_positions=1_000_000, val_path=None, val_fraction=.1, midgame_only=False, include_history=False):
    dataset = PGNStreamDataset(pgn_path, min_elo, sample_rate, midgame_only, max_positions, include_history=include_history)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if val_path:
        Path(val_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with Path(output_path).open('w') as out:
        val = Path(val_path).open('w') if val_path else None
        try:
            for item in dataset._parse_pgn():
                target = val if val and int(item['game_id'], 16) / 2**64 < val_fraction else out
                target.write(json.dumps(item) + '\n')
                count += 1
        finally:
            if val:
                val.close()
    print(f'Extracted {count} positions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgn', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--elo-min', type=int, default=1800)
    parser.add_argument('--sample-rate', type=float, default=.1)
    parser.add_argument('--max-positions', type=int, default=1_000_000)
    parser.add_argument('--val-out')
    parser.add_argument('--val-fraction', type=float, default=.1)
    parser.add_argument('--history', action='store_true')
    args = parser.parse_args()
    build_jsonl_from_pgn(args.pgn, args.out, args.elo_min, args.sample_rate,
                         args.max_positions, args.val_out, args.val_fraction, include_history=args.history)
