import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import chess
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from chessio.lc0 import encode_history, policy_index
from chessio.records import record_board, holdout_positions
from model.artifacts import artifact_info
from play.mlx import Graph
from train.dataset import ChessPositionDataset
from train.q_rank import ranking_loss


def batch_rows(rows, encode=True, q_values=False):
    boards = [record_board(row) for row in rows]
    inputs = np.stack([encode_history(board) for board in boards]) if encode else None
    mask, target = np.zeros((len(rows), 1858), bool), np.zeros((len(rows), 1858), np.float32)
    if q_values:
        q_target, q_mask = np.zeros_like(target), np.zeros_like(mask)
    for i, (row, board) in enumerate(zip(rows, boards)):
        for move in board.legal_moves:
            mask[i, policy_index(move, board)] = True
        distribution = row['move_probs'] if 'move_probs' in row else {row['move']: 1.}
        for uci, probability in distribution.items():
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves or not np.isfinite(probability) or probability < 0:
                raise ValueError('Invalid training policy')
            target[i, policy_index(move, board)] = probability
        if target[i].sum() <= 0:
            raise ValueError('Empty training policy')
        target[i] /= target[i].sum()
        if q_values:
            for uci, value in row.get('q_values', {}).items():
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves or not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Q labels must be legal, finite and in [0, 1]')
                index = policy_index(move, board)
                q_target[i, index], q_mask[i, index] = value, True
    result = (mx.array(inputs) if encode else None, mx.array(mask), mx.array(target))
    return (*result, mx.array(q_target), mx.array(q_mask)) if q_values else result


def drop_history(inputs, rows, selected):
    result = np.array(inputs)
    for index in np.flatnonzero(selected):
        result[index] = encode_history(chess.Board(rows[index]['fen']))
    return mx.array(result)


def teacher_fractions(rows, default=.75, tactical=.25):
    return mx.array([tactical if row.get('source') in ('lichess-puzzle-stockfish-40-meganodes', 'lichess-live-stockfish-hard-feedback') else default
                     for row in rows], dtype=mx.float32)


def losses(graph, weights, inputs, mask, target, teacher_policy, teacher_wdl, value_weight=.25, teacher_weight=.75, sample_weight=None, amp=False, value_target=None, value_mask=None, conditional_policy=False, q_target=None, q_mask=None, q_value_weight=0., q_objective='calibration'):
    if amp:
        weights = {name: value.astype(mx.float16) for name, value in weights.items()}
    policy, value_logits = graph.forward(inputs, weights, raw_value=True)
    policy, value_logits = policy.astype(mx.float32), value_logits.astype(mx.float32)
    q_loss = mx.array(0.)
    if q_value_weight:
        if q_target is None or q_mask is None:
            raise ValueError('Q supervision requires explicit targets and masks')
        terms = mx.logaddexp(mx.array(0., dtype=mx.float32), policy) - policy * q_target
        counts = q_mask.sum(-1)
        rows = (terms * q_mask).sum(-1) / mx.maximum(counts, 1)
        quality = mx.ones((inputs.shape[0],)) if sample_weight is None else sample_weight
        labelled = quality * (counts > 0)
        q_loss = (rows * labelled).sum() / mx.maximum(labelled.sum(), 1e-8)
        if q_objective == 'ranking':
            q_loss = ranking_loss(policy, q_target, q_mask, quality)
    policy = mx.where(mask, policy, -1e9)
    log_probs = policy - mx.logsumexp(policy, axis=-1, keepdims=True)
    quality = mx.ones((inputs.shape[0],)) if sample_weight is None else sample_weight
    quality = quality / quality.sum()
    label_log_probs = log_probs
    if conditional_policy:
        support = target > 0
        partial = (support.sum(-1) > 1) & (support.sum(-1) < mask.sum(-1))
        label_logits = mx.where(mx.where(partial[:, None], support, mask), policy, -1e9)
        label_log_probs = label_logits - mx.logsumexp(label_logits, axis=-1, keepdims=True)
    policy_rows = -(target * label_log_probs).sum(-1)
    distill_rows = -(teacher_policy * log_probs).sum(-1)
    policy_loss = (policy_rows * quality).sum()
    distill_loss = (distill_rows * quality).sum()
    value_loss = mx.array(0.)
    if value_weight:
        value_log_probs = value_logits - mx.logsumexp(value_logits, axis=-1, keepdims=True)
        wdl = mx.exp(value_log_probs)
        value_loss = (-(teacher_wdl * value_log_probs).sum(-1) * quality).sum()
        value_loss += (mx.square(wdl[:, 0] - wdl[:, 2] - teacher_wdl[:, 0] + teacher_wdl[:, 2]) * quality).sum()
        if value_target is not None:
            value_quality = quality * value_mask
            value_loss = (mx.square(wdl[:, 0] - wdl[:, 2] - value_target) * value_quality).sum() / mx.maximum(value_quality.sum(), 1e-8)
    blended_policy = (((1 - teacher_weight) * policy_rows + teacher_weight * distill_rows) * quality).sum()
    result = (blended_policy + value_weight * value_loss + q_value_weight * q_loss, policy_loss, distill_loss, value_loss)
    return (*result, q_loss) if q_value_weight else result


def cache_teacher(teacher, dataset, path, batch_size, identity):
    path = Path(path)
    metadata_path = path.with_suffix('.json')
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata['identity'] == identity and metadata['artifact'] == artifact_info(path):
            return np.load(path, mmap_mode='r')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npy')
    cache = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16, shape=(len(dataset), 1861))
    for start in range(0, len(dataset), batch_size):
        rows = [dataset.read_row(i) for i in range(start, min(start + batch_size, len(dataset)))]
        inputs, _, _ = batch_rows(rows)
        policy, wdl = teacher.run(inputs.astype(mx.float16), teacher.weights)
        values = np.concatenate((np.array(policy), np.array(wdl)), axis=-1).astype(np.float16)
        if not np.isfinite(values).all():
            raise FloatingPointError('Non-finite teacher cache')
        cache[start:start + len(rows)] = values
        if start % (batch_size * 50) == 0:
            print(json.dumps({'teacher_cached': start + len(rows), 'total': len(dataset)}), flush=True)
    cache.flush()
    del cache
    temporary.replace(path)
    metadata_path.write_text(json.dumps({'identity': identity, 'artifact': artifact_info(path)}, indent=2) + '\n')
    return np.load(path, mmap_mode='r')


def finite_gradients(gradient, parameters, arguments, scale, amp):
    retries = 0
    while True:
        values, grads = gradient(parameters, *arguments, mx.array(scale, dtype=mx.float32))
        values = (values[0] / scale, *values[1:])
        grads = {name: value / scale for name, value in grads.items()}
        norm = mx.sqrt(sum(mx.sum(mx.square(value)) for value in grads.values()))
        mx.eval(values, norm)
        if not all(np.isfinite(float(value)) for value in values):
            raise FloatingPointError('Non-finite MLX forward loss')
        if np.isfinite(float(norm)):
            return values, grads, norm, scale, retries
        if not amp or scale <= 1:
            raise FloatingPointError('Non-finite MLX gradient at minimum loss scale')
        scale, retries = max(1, scale / 2), retries + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student', required=True)
    parser.add_argument('--teacher')
    parser.add_argument('--data', required=True)
    parser.add_argument('--exclude-data', help='Comma-separated frozen JSONL tests; exclude their complete position lines')
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--amp', choices=['fp16', 'none'], default='fp16')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--checkpoint-every', type=int, default=0, help='Save current weights after this many training samples; zero saves only complete epochs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--teacher-cache', help='Memory-mapped teacher outputs, shared across epochs')
    parser.add_argument('--feature-cache', help='Cache the frozen policy backbone once; requires the original student as teacher and no history dropout')
    parser.add_argument('--conditional-policy', action='store_true', help='Fit partial multi-PV labels within their recorded support; keep the original teacher over all legal moves')
    parser.add_argument('--q-value-weight', type=float, default=0., help='Fit sigmoid policy scores to the recorded legal-action Q targets')
    parser.add_argument('--q-objective', choices=['calibration', 'ranking'], default='calibration')
    parser.add_argument('--tune-last-blocks', type=int, default=0, help='Also fit this many final transformer blocks in policy mode')
    parser.add_argument('--teacher-temperature', type=float, default=.25)
    parser.add_argument('--teacher-weight', type=float, default=.75, help='Policy distillation fraction; use a lower weight for verified tactical labels')
    parser.add_argument('--tactical-teacher-weight', type=float, default=.25)
    parser.add_argument('--history-dropout', type=float, default=0., help='Train the student on FEN-only inputs for this fraction; teacher retains history')
    parser.add_argument('--mode', choices=['distill', 'policy', 'expert'], default='distill', help='Distill the network, fit policy heads, or fit the network to expert moves')
    args = parser.parse_args()
    if not 0 <= args.teacher_weight <= 1 or not 0 <= args.tactical_teacher_weight <= 1:
        parser.error('Teacher weight must be in [0, 1]')
    if not 0 <= args.history_dropout <= 1:
        parser.error('History dropout must be in [0, 1]')
    if args.feature_cache and (args.mode != 'policy' or not args.teacher or args.history_dropout or args.teacher_cache or args.amp != 'fp16'):
        parser.error('Feature caching requires policy mode, an FP16 teacher, FP16 compute, no history dropout and no separate teacher cache')
    if args.conditional_policy and (args.mode not in ('policy', 'distill') or not args.teacher or args.teacher_weight <= 0):
        parser.error('Conditional policies require policy or distillation mode and a positive teacher weight')
    if not np.isfinite(args.q_value_weight) or args.q_value_weight < 0:
        parser.error('Q weight must be finite and nonnegative')
    if args.checkpoint_every < 0:
        parser.error('Checkpoint interval must be nonnegative')
    if args.tune_last_blocks < 0 or (args.tune_last_blocks and args.mode != 'policy'):
        parser.error('Final-block tuning requires policy mode and a nonnegative block count')
    if args.epochs <= 0 or args.batch_size <= 0 or not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.teacher_temperature) or args.teacher_temperature <= 0:
        parser.error('Positive epochs, batch size and learning rate required')
    graph = Graph(args.student)
    source_metadata = json.loads((Path(args.student) / 'metadata.json').read_text())
    if args.mode == 'distill' and not args.teacher:
        parser.error('Distillation requires --teacher')
    teacher = Graph(args.teacher) if args.teacher and args.mode in ('distill', 'policy') else None
    if graph.quantized:
        parser.error('Train the unquantized student, then quantize the result')
    weights = dict(graph.weights)
    for name in list(weights):
        if name.endswith('.left'):
            right = name[:-5] + '.right'
            weights[name], weights[right] = weights[name].astype(mx.float32), weights[right].astype(mx.float32)
            left_norm = mx.sqrt(mx.square(weights[name]).sum(0))
            right_norm = mx.sqrt(mx.square(weights[right]).sum(1))
            factor = mx.sqrt(mx.maximum(right_norm, 1e-8) / mx.maximum(left_norm, 1e-8))
            weights[name], weights[right] = weights[name] * factor, weights[right] / factor[:, None]
    matrix_names = {node[1][1] for node in graph.nodes if node[0] == 'MatMul' and node[1][1] in weights}
    blocks = sorted({int(match[1]) for name in weights if (match := re.match(r'^/encoder(\d+)/', name))})
    if args.tune_last_blocks > len(blocks):
        parser.error('Requested more final transformer blocks than this graph contains')
    prefixes = ('/policy/', *(f'/encoder{i}/' for i in blocks[-args.tune_last_blocks:])) if args.tune_last_blocks else ('/policy/',)
    parameters = {name: value.astype(mx.float32) for name, value in weights.items() if (not name.startswith('/const/') or name in matrix_names) and value.size > 1
                  and name not in source_metadata.get('frozen_weights', ())
                  and (args.mode != 'policy' or name.startswith(prefixes))}
    if not parameters:
        parser.error('No trainable weights for this mode')
    frozen = {name: value for name, value in weights.items() if name not in parameters}
    excluded = args.exclude_data.split(',') if args.exclude_data else []
    dataset = ChessPositionDataset(args.data, augment=False, max_samples=args.max_samples,
                                   exclude=holdout_positions(excluded))
    if not len(dataset):
        parser.error('Empty training dataset')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(args.student) / 'graph.json', out / 'graph.json')
    metadata = dict(source_metadata)
    metadata.update(student_source=artifact_info(args.student), teacher_source=artifact_info(args.teacher) if teacher else None,
                    training_data=artifact_info(args.data), seed=args.seed, learning_rate=args.lr,
                    training_epochs=args.epochs, training_samples=len(dataset), trainable_scalars=sum(p.size for p in parameters.values()),
                    training_epochs_completed=0, training_samples_processed=0,
                    objective=f'Quality-weighted {1-args.teacher_weight:g} label policy, {args.teacher_weight:g} teacher policy, 0.25 teacher WDL CE plus centred-value MSE',
                    training_device=str(mx.default_device()), teacher_temperature=args.teacher_temperature, mode=args.mode,
                    master_dtype='float32', compute_dtype='float16' if args.amp == 'fp16' else 'float32',
                    loss_scale=1024 if args.amp == 'fp16' else 1)
    metadata['history_dropout'] = args.history_dropout
    metadata['excluded_data'] = {path: artifact_info(path) for path in excluded}
    metadata['heldout_excluded'] = dataset.excluded
    metadata['loss_scale_policy'] = 'Halve and retry the unchanged batch on FP16 gradient overflow, before any optimizer update'
    metadata['optimizer_bias_correction'] = True
    metadata['tactical_teacher_weight'] = args.tactical_teacher_weight
    if args.amp == 'none' and any(node[0] == 'Conv' for node in graph.nodes):
        metadata['inference_dtype'] = 'float32'
    metadata['objective'] += f'; verified tactical rows use {args.tactical_teacher_weight:g} teacher policy'
    metadata['runtime_source'] = {path: artifact_info(path)['sha256'] for path in
        ('train/mlx.py', 'train/q_rank.py', 'play/mlx.py', 'chessio/lc0.py', 'chessio/records.py', 'train/dataset.py')}
    metadata['q_objective'] = args.q_objective
    if args.mode == 'policy':
        metadata['objective'] = (f'Policy heads only: {1-args.teacher_weight:g} label CE, {args.teacher_weight:g} frozen teacher CE'
                                 if teacher else 'Legal-move cross entropy on engine labels; policy heads only')
        metadata['trainable_prefixes'] = list(prefixes)
        metadata['trainable_tensors'] = sorted(parameters)
        if args.tune_last_blocks:
            metadata['objective'] += f'; also tune the final {args.tune_last_blocks} transformer blocks'
        metadata['objective'] += f'; verified tactical rows use {args.tactical_teacher_weight:g} teacher policy'
    elif args.mode == 'expert':
        metadata['objective'] = 'Quality-weighted legal-move cross entropy plus masked centred-value MSE where a real value label is present; whole network'
    metadata['conditional_policy'] = args.conditional_policy
    if args.conditional_policy:
        metadata['objective'] += '; partial multi-PV label CE is conditional on positive recorded support; single-action and full-support labels retain full legal CE'
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    cache_identity = {'teacher': metadata['teacher_source'], 'data': metadata['training_data'],
                      'excluded_data': metadata['excluded_data'], 'samples': len(dataset),
                      'offsets_sha256': hashlib.sha256(dataset.offsets.tobytes()).hexdigest(),
                      'runtime_source': metadata['runtime_source']}
    cached = cache_teacher(teacher, dataset, args.teacher_cache, args.batch_size, cache_identity) if args.teacher_cache and teacher else None
    training_graph, features = graph, None
    if args.feature_cache:
        if metadata['teacher_source'] != metadata['student_source']:
            parser.error('Cached policy features require the unchanged student as teacher')
        if any(mx.issubdtype(value.dtype, mx.floating) and value.dtype != mx.float16 for value in graph.weights.values()):
            parser.error('Cached policy features require FP16 source weights')
        from train.policy_cache import cache_features
        metadata['runtime_source']['train/policy_cache.py'] = artifact_info('train/policy_cache.py')['sha256']
        training_graph, features = cache_features(graph, parameters, dataset, args.feature_cache,
                                                  args.batch_size, cache_identity, batch_rows)
        metadata['feature_cache'] = artifact_info(Path(args.feature_cache).with_suffix('.json'))
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    optimizer = optim.AdamW(args.lr, weight_decay=1e-4, bias_correction=True)
    scale = metadata['loss_scale']
    metadata['q_value_weight'] = args.q_value_weight
    if args.q_value_weight:
        metadata['q_prediction'] = ('Relative policy logits; regret-weighted ranking over known Q labels, no absolute probability calibration'
                                    if args.q_objective == 'ranking' else 'sigmoid(policy_logits); only explicitly recorded legal actions contribute to the Q loss')
        metadata['objective'] += f'; {args.q_value_weight:g} masked legal-action Q {args.q_objective}'
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    def objective(p, x, mask, target, teacher_policy, teacher_wdl, quality, teacher_fraction, value_target, value_mask, q_target, q_mask, loss_scale):
        values = losses(training_graph, {**frozen, **p}, x, mask, target, teacher_policy, teacher_wdl,
                        value_weight=.25 if args.mode != 'policy' and (teacher or args.mode == 'expert') else 0., teacher_weight=teacher_fraction,
                        sample_weight=quality, amp=args.amp == 'fp16',
                        value_target=value_target if args.mode == 'expert' else None, value_mask=value_mask,
                        conditional_policy=args.conditional_policy, q_target=q_target, q_mask=q_mask,
                        q_value_weight=args.q_value_weight, q_objective=args.q_objective)
        return (values[0] * loss_scale, *values[1:])
    gradient = mx.compile(mx.value_and_grad(objective))
    rng = np.random.default_rng(args.seed)
    metric_names = ('loss', 'policy_loss', 'distill_loss', 'value_loss') + (('q_loss',) if args.q_value_weight else ())
    def save_weights(completed_epochs, samples):
        output_weights = {**{name: mx.array(value) for name, value in graph.constants.items()},
                          **{name: value.astype(mx.float16) for name, value in {**frozen, **parameters}.items()}}
        temporary = out / 'weights.tmp.safetensors'
        mx.save_safetensors(str(temporary), output_weights)
        temporary.replace(out / 'weights.safetensors')
        metadata.update(training_epochs_completed=completed_epochs, training_samples_processed=samples)
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    with (out / 'train.jsonl').open('w') as log:
        for epoch in range(args.epochs):
            start, total, count, overflow_retries = time.perf_counter(), np.zeros(len(metric_names)), 0, 0
            checkpoint_at = args.checkpoint_every
            indices = rng.permutation(len(dataset))
            for offset in range(0, len(indices), args.batch_size):
                rows = [dataset.read_row(int(i)) for i in indices[offset:offset + args.batch_size]]
                quality = np.array([row.get('weight', 1.) for row in rows], dtype=np.float32)
                if not np.isfinite(quality).all() or (quality <= 0).any():
                    raise ValueError('Training weights must be finite and positive')
                batch = batch_rows(rows, encode=features is None, q_values=bool(args.q_value_weight))
                if args.q_value_weight:
                    inputs, mask, target, q_target, q_mask = batch
                else:
                    inputs, mask, target = batch
                    q_target = q_mask = None
                if features is not None:
                    selected = indices[offset:offset + args.batch_size]
                    inputs = mx.array(np.asarray(features[0][selected]).reshape(-1, *features[0].shape[2:]))
                inputs = inputs.astype(mx.float16)
                if features is not None:
                    teacher_logits = mx.array(np.asarray(features[1][selected]))
                    teacher_wdl = mx.zeros((len(rows), 3))
                elif teacher is None:
                    teacher_logits, teacher_wdl = None, mx.zeros((len(rows), 3))
                elif cached is None:
                    teacher_logits, teacher_wdl = teacher.run(inputs.astype(mx.float16), teacher.weights)
                else:
                    values = mx.array(np.asarray(cached[indices[offset:offset + args.batch_size]]))
                    teacher_logits, teacher_wdl = values[:, :1858], values[:, 1858:]
                teacher_policy = mx.softmax(mx.where(mask, teacher_logits.astype(mx.float32) / args.teacher_temperature, -1e9), axis=-1) if teacher else target
                if args.history_dropout:
                    inputs = drop_history(inputs, rows, rng.random(len(rows)) < args.history_dropout)
                teacher_fraction = teacher_fractions(rows, args.teacher_weight, args.tactical_teacher_weight)
                value_target = np.array([row.get('value', 0.) for row in rows], dtype=np.float32)
                if not np.isfinite(value_target).all() or (np.abs(value_target) > 1).any():
                    raise ValueError('Value labels must be finite and in [-1, 1]')
                value_mask = mx.array([float('value' in row) for row in rows])
                values, grads, norm, scale, retries = finite_gradients(gradient, parameters,
                    (inputs, mask, target, teacher_policy, teacher_wdl.astype(mx.float32), mx.array(quality), teacher_fraction,
                     mx.array(value_target), value_mask, q_target, q_mask), scale, args.amp == 'fp16')
                overflow_retries += retries
                if retries:
                    print(json.dumps({'epoch': epoch + 1, 'batch_offset': offset, 'overflow_retries': retries, 'loss_scale': scale}), flush=True)
                grads = {name: value / mx.maximum(1., norm) for name, value in grads.items()}
                parameters = optimizer.apply_gradients(grads, parameters)
                mx.eval(parameters, optimizer.state, values, norm)
                total += np.array([float(value) for value in values]) * len(rows)
                count += len(rows)
                if args.checkpoint_every and count >= checkpoint_at:
                    save_weights(epoch, epoch * len(dataset) + count)
                    checkpoint_at = count + args.checkpoint_every
                if offset % (args.batch_size * 50) == 0:
                    print(json.dumps({'epoch': epoch + 1, 'samples': count, 'loss': float(values[0]), 'gradient_norm': float(norm), 'loss_scale': scale}), flush=True)
            metrics = dict(zip(metric_names, (total / count).tolist()))
            metrics.update(epoch=epoch + 1, samples=count, seconds=time.perf_counter() - start,
                           loss_scale_final=scale, overflow_retries=overflow_retries)
            log.write(json.dumps(metrics) + '\n')
            log.flush()
            save_weights(epoch + 1, (epoch + 1) * len(dataset))
            print(json.dumps(metrics), flush=True)
    if artifact_info(args.data) != metadata['training_data']:
        raise ValueError('Training data changed during the run')


if __name__ == '__main__':
    main()
