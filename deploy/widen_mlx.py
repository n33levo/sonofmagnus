import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from model.artifacts import artifact_info


def widen(source, output, channels=160, seed=0):
    source, output = Path(source), Path(output)
    graph = json.loads((source / 'graph.json').read_text())
    if graph.get('quantized') or any(node[0] in ('Attention', 'LayerNormalization') for node in graph['nodes']):
        raise ValueError('Widen an unquantized residual CNN')
    kernels = {node[1][1] for node in graph['nodes'] if node[0] == 'Conv'}
    if not kernels or any(node[3].get('group', 1) != 1 for node in graph['nodes'] if node[0] == 'Conv'):
        raise ValueError('Expected ordinary residual convolutions')
    weights = mx.load(str(source / 'weights.safetensors'))
    first = next(node[1][1] for node in graph['nodes'] if node[0] == 'Conv')
    width = weights[first].shape[0]
    if channels <= width:
        raise ValueError('The new width must exceed the source width')
    shapes = {node[1][1] for node in graph['nodes'] if node[0] in ('Reshape', 'Split') and len(node[1]) > 1}
    rng, expanded = np.random.default_rng(seed), {}
    for name, tensor in weights.items():
        value = np.array(tensor)
        if name in graph['integer_constants']:
            if name in shapes:
                value = np.where(value == width, channels, np.where(value == 2 * width, 2 * channels, value))
            expanded[name] = mx.array(value)
            continue
        new_shape, indices = [], []
        for dimension in value.shape:
            split_gate = '/se/' in name and dimension == 2 * width
            new_shape.append(2 * channels if split_gate else channels if dimension == width else dimension)
            indices.append(np.r_[np.arange(width), channels + np.arange(width)] if split_gate else np.arange(dimension))
        result = np.zeros(new_shape, dtype=value.dtype)
        # Added features have random inputs and zero connections into the existing features and heads.
        if name in kernels and value.shape[0] == width:
            result[width:] = rng.normal(0, np.sqrt(2 / np.prod(new_shape[1:])), result[width:].shape)
        elif value.ndim == 2 and value.shape[1] == width and '/se/' not in name:
            result[:, width:] = rng.normal(0, np.sqrt(2 / new_shape[0]), result[:, width:].shape)
        result[np.ix_(*indices)] = value
        expanded[name] = mx.array(result)
    for node in graph['nodes']:
        if node[0] == 'Split' and 'split' in node[3]:
            node[3]['split'] = [channels if size == width else size for size in node[3]['split']]
    count = sum(value.size for name, value in expanded.items() if name not in graph['integer_constants'])
    if count > 6_000_000:
        raise ValueError(f'The widened model exceeds the parameter budget: {count}')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'graph.json').write_text(json.dumps(graph, separators=(',', ':')) + '\n')
    mx.save_safetensors(str(output / 'weights.safetensors'), expanded)
    metadata = json.loads((source / 'metadata.json').read_text())
    metadata.update(widening_source=artifact_info(source), original_channels=width, widened_channels=channels,
                    widening_seed=seed, stored_scalars_before_quantization=count,
                    weights_and_graph_bytes=sum(path.stat().st_size for path in output.iterdir()),
                    initialization='Existing paths unchanged; added features have random inputs and zero outgoing connections')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--channels', type=int, default=160)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(widen(args.model, args.out, args.channels, args.seed), indent=2))
