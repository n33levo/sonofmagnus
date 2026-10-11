import argparse
import json
from pathlib import Path

import numpy as np

from model.artifacts import artifact_info


def piece_geometry():
    squares = np.arange(64)
    df = squares[None, :] % 8 - squares[:, None] % 8
    dr = squares[None, :] // 8 - squares[:, None] // 8
    diagonal = (abs(df) == abs(dr)) & (df != 0)
    straight = ((df == 0) | (dr == 0)) & ((df != 0) | (dr != 0))
    knight = ((abs(df) == 1) & (abs(dr) == 2)) | ((abs(df) == 2) & (abs(dr) == 1))
    king = np.maximum(abs(df), abs(dr)) == 1
    return np.stack([matrix for direction in (1, -1) for matrix in
                     ((abs(df) == 1) & (dr == direction), knight, diagonal, straight, diagonal | straight, king)]).astype(np.float16)


def add_geometry(source, output):
    import mlx.core as mx
    source, output = Path(source), Path(output)
    graph = json.loads((source / 'graph.json').read_text())
    if graph.get('quantized') or graph['nodes'][0][0] != 'Conv':
        raise ValueError('Use an unquantized classical CNN')
    weights = mx.load(str(source / 'weights.safetensors'))
    node = graph['nodes'][0]
    kernel = node[1][1]
    if weights[kernel].shape[1] != 112:
        raise ValueError('Expected classical 112-plane input')
    weights[kernel] = mx.concatenate([weights[kernel], mx.zeros((weights[kernel].shape[0], 12, 3, 3), dtype=weights[kernel].dtype)], axis=1)
    prefix = '/const/geometry/'
    constants = {'start': [0], 'end': [12], 'axis': [1], 'flat': [0, 12, 1, 64], 'planes': [0, 12, 8, 8]}
    for name, value in constants.items():
        weights[prefix + name] = mx.array(value, dtype=mx.int32)
        graph['integer_constants'].append(prefix + name)
    weights[prefix + 'basis'] = mx.array(piece_geometry())
    inputs = graph['inputs'][0]
    nodes = [
        ['Slice', [inputs, prefix + 'start', prefix + 'end', prefix + 'axis'], ['/geometry/pieces'], {}],
        ['Reshape', ['/geometry/pieces', prefix + 'flat'], ['/geometry/flat'], {}],
        ['MatMul', ['/geometry/flat', prefix + 'basis'], ['/geometry/reach'], {}],
        ['Reshape', ['/geometry/reach', prefix + 'planes'], ['/geometry/planes'], {}],
        ['Concat', [inputs, '/geometry/planes'], ['/geometry/input'], {'axis': 1}],
    ]
    node[1][0] = '/geometry/input'
    graph['nodes'] = nodes + graph['nodes']
    metadata = json.loads((source / 'metadata.json').read_text())
    matrices = {node[1][1] for node in graph['nodes'] if node[0] == 'MatMul'}
    floating = {name: value for name, value in weights.items() if mx.issubdtype(value.dtype, mx.floating)}
    metadata.update(source_artifact=artifact_info(source), geometry='Empty-board piece reachability from the current position; sliding blockers are left to the network',
                    frozen_weights=[*metadata.get('frozen_weights', ()), prefix + 'basis'], geometry_planes=12, added_trainable_scalars=12 * weights[kernel].shape[0] * 9,
                    initialization='Zero new input-convolution weights preserve the source function',
                    producer=artifact_info(__file__))
    metadata['stored_scalars_before_quantization'] = sum(value.size for value in floating.values())
    metadata['trainable_scalars'] = sum(value.size for name, value in floating.items() if value.size > 1
                                      and (not name.startswith('/const/') or name in matrices)
                                      and name not in metadata['frozen_weights'])
    metadata['fixed_geometry_scalars'] = weights[prefix + 'basis'].size
    output.mkdir(parents=True, exist_ok=False)
    (output / 'graph.json').write_text(json.dumps(graph, separators=(',', ':')) + '\n')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    mx.save_safetensors(str(output / 'weights.safetensors'), weights)
    return artifact_info(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add fixed piece geometry to a classical history CNN')
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(add_geometry(args.model, args.out), indent=2))
