import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import json
import os
from pathlib import Path
import threading
import time

import chess.engine

from chessio.records import record_board
from model.artifacts import artifact_info
from train.distill_labeler import label_position


def relabel(source, output, engine_path, depth=12, workers=2):
    source, output, engine_path = Path(source), Path(output), Path(engine_path).resolve()
    if source.resolve() == output.resolve() or depth < 1 or workers < 1:
        raise ValueError('Use distinct paths, positive depth and positive workers')
    identity = dict(source=artifact_info(source), engine=artifact_info(engine_path),
                    depth=depth, all_legal_moves=True,
                    producer=artifact_info(__file__), labeler=artifact_info('train/distill_labeler.py'),
                    history_encoder=artifact_info('chessio/records.py'))
    manifest, partial = output.with_suffix('.json'), output.with_suffix(output.suffix + '.part')
    output.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        state = json.loads(manifest.read_text())
        if state['identity'] != identity:
            raise ValueError('Source, engine, code or labeling settings changed')
        if output.exists():
            if state.get('status') == 'running' and not partial.exists():
                output.replace(partial)
            elif state.get('artifact') != artifact_info(output):
                raise ValueError('Completed output differs from its manifest')
            else:
                return state
    else:
        if output.exists() or partial.exists():
            raise ValueError('Existing output has no matching manifest')
        state = dict(identity=identity, status='running')
        manifest.write_text(json.dumps(state, indent=2) + '\n')
    # Only a crash-truncated final row may be discarded; verify every committed prefix row.
    done = 0
    with source.open('rb') as inputs, partial.open('a+b') as previous:
        previous.seek(0)
        while line := previous.readline():
            offset = previous.tell() - len(line)
            if not line.endswith(b'\n'):
                previous.truncate(offset)
                break
            row, original = json.loads(line), json.loads(next(inputs))
            if row['fen'] != original['fen'] or row.get('history') != original.get('history'):
                raise ValueError('Committed labels differ from the input prefix')
            board = record_board(row)
            if set(row['q_values']) != {m.uci() for m in board.legal_moves} or row['label_depth_min'] < depth:
                raise ValueError('Committed labels are incomplete or below requested depth')
            done += 1
    local, engines = threading.local(), []

    def initialize():
        local.engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
        local.engine.configure({'Threads': 1, 'Hash': 128})
        engines.append(local.engine)

    def label(line):
        row = json.loads(line)
        board = record_board(row)
        if not board.is_valid() or board.is_game_over(claim_draw=True):
            raise ValueError('Selected source contains an invalid or terminal board')
        result = label_position(board, local.engine, depth, 0, 5, True)
        if result['label_depth_min'] < depth:
            raise ValueError('Stockfish stopped below requested depth')
        return json.dumps({**row, **result, 'source': 'selfplay-stockfish-all-actions', 'weight': 3.}) + '\n'

    started, initial = time.perf_counter(), done
    try:
        with source.open() as inputs, partial.open('a') as stream, ThreadPoolExecutor(workers, initializer=initialize) as pool:
            for _ in range(done):
                next(inputs)
            while batch := list(islice(inputs, 32 * workers)):
                for row in pool.map(label, batch):
                    stream.write(row)
                    done += 1
                stream.flush()
                os.fsync(stream.fileno())
                print(json.dumps({'labeled': done, 'new_labels': done - initial,
                                  'seconds': time.perf_counter() - started}), flush=True)
        if not done:
            raise ValueError('Empty source')
        if artifact_info(source) != identity['source'] or artifact_info(engine_path) != identity['engine']:
            raise ValueError('Input or engine changed during labeling')
        partial.replace(output)
        state.update(status='complete', positions=done, artifact=artifact_info(output),
                     seconds_this_run=time.perf_counter() - started, engine_name=[e.id for e in engines])
        manifest.write_text(json.dumps(state, indent=2) + '\n')
        return state
    finally:
        for engine in engines:
            engine.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resumable, all-legal-action Stockfish labels')
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(relabel(args.source, args.out, args.engine, args.depth, args.workers), indent=2))
