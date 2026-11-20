"""Extract legal mating lines from the frozen Lichess puzzle archive."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

import chess
import zstandard

from chessio.records import holdout_positions
from model.artifacts import artifact_info


def mating_rows(puzzle):
    board = chess.Board(puzzle['FEN'])
    if not board.is_valid():
        raise ValueError('Invalid puzzle board')
    moves = puzzle['Moves'].split()
    lengths = [n for n in range(1, 6) if f'mateIn{n}' in puzzle['Themes'].split()]
    if len(lengths) != 1 or len(moves) != 2 * lengths[0]:
        raise ValueError('Mate tag and line length disagree')
    root, history, keys, rows = board.fen(), [], set(), []
    solver = not board.turn
    for ply, uci in enumerate(moves):
        keys.update(' '.join(b.fen().split()[:4]) for b in (board, board.mirror()))
        move = board.parse_uci(uci)
        if board.turn == solver:
            rows.append(dict(fen=board.fen(), root_fen=root, history=history.copy(), move=uci,
                game_id='puzzle-' + puzzle['PuzzleId'], source='lichess-puzzle-stockfish-40-meganodes',
                weight=1., mate_moves_remaining=(len(moves)-ply+1)//2,
                puzzle_rating=int(puzzle['Rating']), themes=puzzle['Themes'].split()))
        board.push(move); history.append(uci)
    keys.update(' '.join(b.fen().split()[:4]) for b in (board, board.mirror()))
    if not board.is_checkmate() or board.turn == solver:
        raise ValueError('Puzzle line does not finish with the solver delivering mate')
    return rows, keys


def puzzle_split(puzzle_id, seed):
    digest = hashlib.blake2b(f'{seed}:{puzzle_id}'.encode(), digest_size=8).digest()
    return 'validation' if int.from_bytes(digest, 'little') % 20 == 0 else 'fit'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--exclude-data', required=True, help='Comma-separated held-out and earlier fitting JSONL files')
    parser.add_argument('--out', type=Path, required=True, help='Output prefix')
    parser.add_argument('--fit-per-band', type=int, nargs=4, default=[2000,1600,1250,150])
    parser.add_argument('--validation-per-band', type=int, nargs=4, default=[108,80,64,4])
    parser.add_argument('--seed', type=int, default=2026101733)
    args = parser.parse_args()
    if min(*args.fit_per_band, *args.validation_per_band) < 1:
        parser.error('Positive puzzle counts required')
    outputs = {role: args.out.with_suffix(f'.{role}.jsonl') for role in ('fit','validation')}
    partial = {role: path.with_suffix('.part') for role,path in outputs.items()}
    manifest, progress = args.out.with_suffix('.json'), args.out.with_suffix('.progress.json')
    if any(p.exists() for p in [*outputs.values(),*partial.values(),manifest,progress]):
        raise FileExistsError('Use a new output prefix')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    exclusions = args.exclude_data.split(',')
    inputs = {str(p):artifact_info(p) for p in [args.archive,Path(__file__),*exclusions]}
    blocked, existing_ids = holdout_positions(exclusions), set()
    for path in exclusions:
        for row in map(json.loads,Path(path).open()):
            game_id = row.get('game_id') or ''
            if game_id.startswith('puzzle-'): existing_ids.add(game_id[7:])
    used, counts, bands, positions = set(), Counter(), Counter(), Counter()
    targets = dict(fit=args.fit_per_band,validation=args.validation_per_band)
    streams = {role:path.open('x') for role,path in partial.items()}
    try:
        with args.archive.open('rb') as raw, zstandard.ZstdDecompressor().stream_reader(raw) as decoded, io.TextIOWrapper(decoded) as text:
            for puzzle in csv.DictReader(text):
                counts['scanned'] += 1
                rating = int(puzzle['Rating']); band = (rating-1800)//300
                if not 0 <= band < 4 or not {'mateIn3','mateIn4','mateIn5'} & set(puzzle['Themes'].split()): continue
                if int(puzzle['NbPlays']) < 100 or int(puzzle['Popularity']) < 80: continue
                role = puzzle_split(puzzle['PuzzleId'],args.seed)
                if bands[role,band] >= targets[role][band]: continue
                if puzzle['PuzzleId'] in existing_ids:
                    counts['earlier_puzzle_id'] += 1; continue
                try: rows,keys = mating_rows(puzzle)
                except ValueError:
                    counts['invalid_mating_line'] += 1; continue
                if keys & blocked:
                    counts['excluded_position_or_mirror'] += 1; continue
                if keys & used:
                    counts['duplicate_line_or_mirror'] += 1; continue
                used.update(keys); existing_ids.add(puzzle['PuzzleId']); bands[role,band] += 1
                for row in rows: streams[role].write(json.dumps(row,separators=(',',':'))+'\n')
                positions[role] += len(rows)
                if sum(bands.values()) % 100 == 0:
                    status=dict(counts=counts,puzzles={f'{r}:{1800+300*b}':bands[r,b] for r in targets for b in range(4)},positions=positions)
                    progress.write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
                if all(bands[r,b] == targets[r][b] for r in targets for b in range(4)): break
    finally:
        for stream in streams.values(): stream.close()
    complete = all(bands[r,b] == targets[r][b] for r in targets for b in range(4))
    if not complete:
        raise RuntimeError(f'Archive exhausted before the predeclared band targets: {dict(bands)}')
    assert all(artifact_info(p)==v for p,v in inputs.items())
    for role,path in outputs.items(): partial[role].replace(path)
    report=dict(inputs=inputs,outputs={r:artifact_info(p) for r,p in outputs.items()},source_url='https://database.lichess.org/lichess_db_puzzle.csv.zst',
        license='CC0',seed=args.seed,targets=targets,quality_filters=dict(min_plays=100,min_popularity=80),counts=counts,puzzles={f'{r}:{1800+300*b}':bands[r,b] for r in targets for b in range(4)},positions=positions,
        complete=True,exclusions=exclusions,split='Puzzle-id hash; entire mating line and its mirror are disjoint across splits and exclusions.',
        scope='Legal complete Lichess solution lines ending in checkmate; not a new proof that the mate is forced or shortest. Solver decisions only. No RL or game-strength result.')
    manifest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','exclusions')}),flush=True)


if __name__=='__main__': main()
