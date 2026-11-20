"""Extract complete middlegame action values from a pinned public shard."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import chess
import msgpack
import zstandard

from chessio.records import holdout_positions
from model.artifacts import artifact_info
from scripts.fetch_data import fetch


REVISION = '42862cdee08db45e0e72a743f0076bbaa330c516'
SHARD = 'train-00000-of-01024.msgpack.zst'
SHA256 = '0cfcb156c2a8cf6fbe1764faa0dc23ef3d355510d92b37e7e1c1264987b142fa'
BYTES = 72116154
URL = f'https://huggingface.co/datasets/prdev/chessbench-full-policy-value/resolve/{REVISION}/{SHARD}'
HOLDOUTS = ','.join(f'data/{name}.jsonl' for name in
                    ('puzzles_holdout', 'puzzles_final3000', 'puzzles_depth3000',
                     'stockfish_holdout', 'eval_holdout', 'endgame_holdout'))


def verified_archive(path):
    path = Path(path)
    return fetch(path.name, {'url': URL, 'bytes': BYTES, 'sha256': SHA256}, path.parent)


def mate_order(value):
    if value == '#':
        return (2, 0)
    if value is None:
        return (0, 0)
    if type(value) is not int or value == 0:
        raise ValueError('Invalid mate annotation')
    return (1 if value > 0 else -1, -value)


def extract(archive, out, positions, excluded, min_pieces=8, max_pieces=24, min_legal_moves=1):
    archive = Path(archive)
    archive_info = artifact_info(archive)
    out = Path(out)
    if out.exists() or out.with_suffix('.json').exists() or positions < 1 or min_legal_moves < 1 or not 2 <= min_pieces <= max_pieces <= 32:
        raise ValueError('Use a new output path and positive count')
    frozen = {path: artifact_info(path) for path in excluded}
    blocked = holdout_positions(excluded)
    blocked |= {' '.join(chess.Board(fen).mirror().fen().split()[:4]) for fen in blocked}
    accepted = scanned = incomplete = 0
    seen = set()
    legal_counts = Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + '.part')
    with Path(archive).open('rb') as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream, partial.open('x') as target:
        for item in msgpack.Unpacker(stream, raw=False, max_buffer_size=128 << 20):
            scanned += 1
            board = chess.Board(item['fen'])
            if not board.is_valid() or board.is_game_over(claim_draw=True) or not min_pieces <= len(board.piece_map()) <= max_pieces:
                continue
            key = ' '.join(board.fen().split()[:4])
            if key in blocked or key in seen:
                continue
            moves = item['moves']
            if set(moves) != {move.uci() for move in board.legal_moves}:
                incomplete += 1
                continue
            if len(moves) < min_legal_moves:
                continue
            if any(not isinstance(score['win_prob'], (int, float)) or
                   not math.isfinite(score['win_prob']) or
                   not 0 <= score['win_prob'] <= 1 for score in moves.values()):
                raise ValueError('Invalid action value')
            values = {move: min(1 - 1e-6, max(1e-6, score['win_prob'])) for move, score in moves.items()}
            order = {move: mate_order(score.get('mate')) for move, score in moves.items()}
            for move, score in moves.items():
                mate = score.get('mate')
                if mate is not None and score['win_prob'] != float(mate == '#' or mate > 0):
                    raise ValueError('Mate sign and action value disagree')
                if mate == '#':
                    board.push_uci(move)
                    checkmate = board.is_checkmate()
                    board.pop()
                    if not checkmate:
                        raise ValueError('Immediate mate annotation is false')
            best = max(moves, key=lambda move: (moves[move]['win_prob'], *order[move]))
            target.write(json.dumps({'fen': board.fen(), 'move_probs': {best: 1.},
                                     'q_values': values, 'mate_distances': {m: v['mate'] for m, v in moves.items() if v.get('mate') is not None},
                                     'source': 'chessbench-full-policy-value'},
                                    separators=(',', ':')) + '\n')
            seen.add(key)
            seen.add(' '.join(board.mirror().fen().split()[:4]))
            legal_counts[len(moves)] += 1
            accepted += 1
            if accepted % 5000 == 0:
                print(json.dumps({'accepted': accepted, 'scanned': scanned}), flush=True)
            if accepted == positions:
                break
    if accepted != positions or artifact_info(archive) != archive_info or any(artifact_info(path) != value for path, value in frozen.items()):
        raise ValueError('Incomplete extraction or changed holdout')
    partial.replace(out)
    result = {'positions': accepted, 'scanned': scanned, 'incomplete_filtered': incomplete,
              'piece_range': [min_pieces, max_pieces],
              'source': URL if archive_info == {'model_bytes': BYTES, 'sha256': SHA256} else None,
              'source_archive': archive_info, 'source_sha256': archive_info['sha256'],
              'q_logit_scale': 1 / .00368208, 'history': 'FEN only; no played game history supplied',
              'q_scale_basis': 'Inferred integer-centipawn logistic grid; upstream card omits calibration',
              'stored_q_bounds': [1e-6, 1-1e-6],
              'mate_ties': 'Use raw unrounded Q, then immediate mate, shortest winning mate, or longest forced loss',
              'min_legal_moves': min_legal_moves, 'legal_action_counts': dict(sorted(legal_counts.items())),
              'excluded': frozen, 'artifact': artifact_info(out),
              'producer': artifact_info(__file__)}
    out.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--positions', type=int, default=50000)
    ap.add_argument('--archive', default='data/chessbench_fullpolicy_00000.msgpack.zst')
    ap.add_argument('--exclude-data', default=HOLDOUTS)
    ap.add_argument('--min-legal-moves', type=int, default=1)
    args = ap.parse_args()
    print(json.dumps(extract(verified_archive(args.archive), args.out, args.positions,
                             args.exclude_data.split(','), min_legal_moves=args.min_legal_moves), indent=2))
