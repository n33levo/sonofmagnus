import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from chessio.records import holdout_positions
from model.artifacts import artifact_info
from play.mlx import Graph
from train.dataset import ChessPositionDataset
from train.mlx import batch_rows


def extend(teacher_path, old_data, data, old_cache, output, excluded=(), batch_size=64):
    old_cache, output = Path(old_cache), Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Refusing to replace an existing cache')
    previous = json.loads(old_cache.with_suffix('.json').read_text())
    identity = previous['identity']
    data_identity = artifact_info(data)
    if previous['artifact'] != artifact_info(old_cache) or identity['teacher'] != artifact_info(teacher_path):
        raise ValueError('Teacher or cached outputs changed')
    if identity['data'] != artifact_info(old_data):
        raise ValueError('Cached training data changed')
    blocked = holdout_positions(excluded)
    source = ChessPositionDataset(old_data, augment=False, exclude=blocked)
    target = ChessPositionDataset(data, augment=False, exclude=blocked)
    exclusions = {str(p): artifact_info(p) for p in excluded}
    runtime = {p: artifact_info(p)['sha256'] for p in identity['runtime_source']}
    if identity['excluded_data'] != exclusions or identity['runtime_source'] != runtime:
        raise ValueError('Cache exclusions or runtime changed')
    if identity['samples'] != len(source) or identity['offsets_sha256'] != hashlib.sha256(source.offsets.tobytes()).hexdigest():
        raise ValueError('Cached sample ordering changed')
    if len(target) < len(source):
        raise ValueError('Continuation is shorter than the cached corpus')
    for i in range(len(source)):
        a, b = source.read_row(i), target.read_row(i)
        if any(a.get(key) != b.get(key) for key in ('fen', 'root_fen', 'history')):
            raise ValueError(f'Teacher input changed at retained row {i}')
    values = np.load(old_cache, mmap_mode='r')
    if values.shape != (len(source), 1861) or values.dtype != np.float16:
        raise ValueError('Unexpected teacher cache shape or dtype')
    # Recompute the old partial batch together with new rows, preserving batch boundaries.
    reused = len(source) // batch_size * batch_size
    teacher = Graph(teacher_path)
    for start in sorted({0, max(0, reused - batch_size)} if reused else set()):
        inputs, _, _ = batch_rows([source.read_row(i) for i in range(start, start + batch_size)])
        policy, wdl = teacher.run(inputs.astype(mx.float16), teacher.weights)
        direct = np.concatenate((np.array(policy), np.array(wdl)), axis=-1).astype(np.float16)
        if not np.array_equal(direct, values[start:start + batch_size]):
            raise ValueError('Reused outputs disagree with the current teacher and batch boundaries')
    temporary = output.with_suffix('.tmp.npy')
    if temporary.exists():
        raise ValueError('An unfinished cache needs review')
    cache = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16, shape=(len(target), 1861))
    for start in range(0, reused, 4096):
        block = values[start:min(start + 4096, reused)]
        if not np.isfinite(block).all():
            raise FloatingPointError('Non-finite reused teacher outputs')
        cache[start:start + len(block)] = block
    for start in range(reused, len(target), batch_size):
        rows = [target.read_row(i) for i in range(start, min(start + batch_size, len(target)))]
        inputs, _, _ = batch_rows(rows)
        policy, wdl = teacher.run(inputs.astype(mx.float16), teacher.weights)
        block = np.concatenate((np.array(policy), np.array(wdl)), axis=-1).astype(np.float16)
        if not np.isfinite(block).all():
            raise FloatingPointError('Non-finite new teacher outputs')
        cache[start:start + len(rows)] = block
        if (start - reused) % (batch_size * 50) == 0:
            print(json.dumps({'teacher_cached': start + len(rows), 'reused': reused, 'total': len(target)}), flush=True)
    cache.flush()
    del cache
    if artifact_info(data) != data_identity or artifact_info(old_data) != identity['data'] or artifact_info(old_cache) != previous['artifact'] or artifact_info(teacher_path) != identity['teacher']:
        raise ValueError('Cache inputs changed during extension')
    temporary.replace(output)
    updated = {**identity, 'data': data_identity, 'samples': len(target),
               'offsets_sha256': hashlib.sha256(target.offsets.tobytes()).hexdigest()}
    manifest = {'identity': updated, 'artifact': artifact_info(output), 'reused_rows': reused,
                'verified_input_prefix_rows': len(source), 'parent_cache': previous['artifact'],
                'producer': artifact_info(__file__)}
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('teacher', 'old-data', 'data', 'old-cache', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--exclude-data', default='')
    args = parser.parse_args()
    extend(args.teacher, args.old_data, args.data, args.old_cache, args.out,
           args.exclude_data.split(',') if args.exclude_data else [])
