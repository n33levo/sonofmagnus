import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def quantize_weights(weights, matrices, bits):
    import mlx.core as mx
    quantized = {}
    if bits == 16:
        return quantized
    if bits not in (4, 6, 8):
        raise ValueError('Supported weight widths: 4, 6, 8, 16')
    for name in sorted(matrices):
        weight = weights[name]
        if weight.ndim != 2 or weight.shape[0] % 32 or weight.shape[1] < 32:
            continue
        packed, scales, biases = mx.quantize(weight.T, group_size=32, bits=bits)
        weights.pop(name)
        weights.update({name + '.packed': packed, name + '.scales': scales, name + '.biases': biases})
        quantized[name] = [32, bits]
    return quantized


def factorize(weight, rank, gram=None, ridge=.01):
    weight = weight.astype(np.float64)
    whitening = None
    if gram is not None:
        gram = np.asarray(gram, dtype=np.float64)
        if gram.shape != (weight.shape[0], weight.shape[0]) or not np.isfinite(gram).all():
            raise ValueError('Invalid activation covariance')
        covariance = (gram + gram.T) / (2 * max(float(np.trace(gram)) / len(gram), 1e-12))
        whitening = np.linalg.cholesky(covariance + ridge * np.eye(len(gram)))
    matrix = weight if whitening is None else whitening.T @ weight
    u, singular, v = np.linalg.svd(matrix, full_matrices=False)
    root = np.sqrt(singular[:rank])
    left, right = u[:, :rank] * root, root[:, None] * v[:rank]
    if whitening is not None:
        left = np.linalg.solve(whitening.T, left)
        scale = np.sqrt(np.maximum(np.linalg.norm(right, axis=1), 1e-12) / np.maximum(np.linalg.norm(left, axis=0), 1e-12))
        left, right = left * scale, right / scale[:, None]
    return left, right, float((singular[rank:] ** 2).sum()), float((singular ** 2).sum())


def export_graph(path, directory, bits=16, source_weights=None, rank=None, calibration=None):
    import mlx.core as mx
    if calibration and rank is None:
        raise ValueError('Activation calibration requires a low-rank export')
    if source_weights:
        from lczero.backends import Weights
        if Weights(str(source_weights)).input_format() != 1:
            raise ValueError('Only classical LC0 input is supported')
    model = onnx.load(path)
    if model.graph.input[0].type.tensor_type.shape.dim[1].dim_value != 112:
        raise ValueError('This exporter supports classical LC0 attention networks')
    outputs = [n.name for n in model.graph.output if n.name.endswith(('/policy', '/wdl'))]
    if len(outputs) != 2:
        raise ValueError('Expected policy and WDL outputs')
    needed, nodes = set(outputs), []
    for node in reversed(model.graph.node):
        if needed.intersection(node.output):
            nodes.append(node)
            needed.update(node.input)
    nodes.reverse()
    supported = {'MatMul', 'Add', 'Mul', 'Reshape', 'Transpose', 'Shape', 'Slice', 'Concat',
                 'Expand', 'Sigmoid', 'Softplus', 'Tanh', 'Relu', 'LayerNormalization', 'Softmax', 'Gather', 'Split'}
    if missing := {node.op_type for node in nodes} - supported:
        raise ValueError(f'Unsupported MLX operations: {missing}')
    values = {w.name: numpy_helper.to_array(w) for w in model.graph.initializer if w.name in needed}
    low_rank, discarded_energy, total_energy = 0, 0., 0.
    moments = mx.load(str(calibration)) if calibration else {}
    if rank is not None:
        if rank <= 0 or rank % 32:
            raise ValueError('Low-rank width must be a positive multiple of 32')
        rewritten = []
        for node in nodes:
            weight = values.get(node.input[1]) if node.op_type == 'MatMul' else None
            if weight is not None and weight.ndim == 2 and rank * sum(weight.shape) < weight.size:
                name = node.input[1]
                if name + '.left' not in values:
                    left, right, error, energy = factorize(weight, rank, moments.get(name))
                    values[name + '.left'], values[name + '.right'] = left.astype(weight.dtype), right.astype(weight.dtype)
                    low_rank += 1
                    discarded_energy += error
                    total_energy += energy
                middle = node.output[0] + '/low_rank'
                rewritten.extend([helper.make_node('MatMul', [node.input[0], name + '.left'], [middle]),
                                  helper.make_node('MatMul', [middle, name + '.right'], list(node.output))])
            else:
                rewritten.append(node)
        nodes = rewritten
        referenced = {name for node in nodes for name in node.input}
        values = {name: value for name, value in values.items() if name in referenced}
    integer = [name for name, value in values.items() if np.issubdtype(value.dtype, np.integer)]
    weights = {name: mx.array(value.astype(np.int32) if name in integer else value) for name, value in values.items()}
    matrices = {node.input[1] for node in nodes if node.op_type == 'MatMul' and node.input[1] in values}
    quantized = quantize_weights(weights, matrices, bits)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    graph = {'inputs': [v.name for v in model.graph.input], 'outputs': outputs, 'integer_constants': integer,
             'quantized': quantized, 'nodes': [[n.op_type, list(n.input), list(n.output),
                {a.name: helper.get_attribute_value(a) for a in n.attribute}] for n in nodes]}
    (directory / 'graph.json').write_text(json.dumps(graph, separators=(',', ':')) + '\n')
    mx.save_safetensors(str(directory / 'weights.safetensors'), weights)
    with open(path, 'rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    stats = {'source_onnx_sha256': digest, 'weight_bits': bits,
             'stored_scalars_before_quantization': sum(v.size for k, v in values.items() if k not in integer),
             'weights_and_graph_bytes': sum(p.stat().st_size for p in directory.iterdir() if p.name != 'metadata.json'), 'quantized_matrices': len(quantized)}
    if rank is not None:
        stats.update(low_rank_width=rank, factorized_matrices=low_rank,
                     relative_squared_weight_error=discarded_energy / total_energy if total_energy else 0.,
                     compression_metric='ridge-regularized activation error' if calibration else 'weight Frobenius error')
    if calibration:
        from model.artifacts import artifact_info
        stats['calibration'] = artifact_info(calibration)
    if source_weights:
        with open(source_weights, 'rb') as stream:
            stats['source_weights_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (directory / 'metadata.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--bits', type=int, choices=[4, 6, 8, 16], default=16)
    parser.add_argument('--source-weights', required=True)
    parser.add_argument('--rank', type=int, help='SVD bottleneck width; requires retraining and new strength tests')
    parser.add_argument('--calibration', help='Input second moments from deploy.calibrate_mlx')
    args = parser.parse_args()
    export_graph(args.onnx, args.out, args.bits, args.source_weights, args.rank, args.calibration)


if __name__ == '__main__':
    main()
