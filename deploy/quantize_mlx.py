import argparse
import json
from pathlib import Path

import mlx.core as mx

from deploy.export_mlx import quantize_weights
from model.artifacts import artifact_info


def quantize(directory, output, bits=8, group_size=32):
    directory, output = Path(directory), Path(output)
    graph = json.loads((directory / 'graph.json').read_text())
    if graph.get('quantized'):
        raise ValueError('Quantize an unquantized source artifact')
    weights = mx.load(str(directory / 'weights.safetensors'))
    matrices = {node[1][1] for node in graph['nodes'] if node[0] == 'MatMul' and node[1][1] in weights}
    graph['quantized'] = quantize_weights(weights, matrices, bits, group_size)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'graph.json').write_text(json.dumps(graph, separators=(',', ':')) + '\n')
    mx.save_safetensors(str(output / 'weights.safetensors'), weights)
    metadata = json.loads((directory / 'metadata.json').read_text())
    metadata.update(weight_bits=bits, quantization_group_size=group_size, quantized_matrices=len(graph['quantized']), source_artifact=artifact_info(directory),
                    weights_and_graph_bytes=sum(p.stat().st_size for p in output.iterdir()))
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    result = artifact_info(output)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--bits', type=int, choices=[2, 3, 4, 6, 8], default=8)
    parser.add_argument('--group-size', type=int, choices=[32, 64, 128], default=32)
    args = parser.parse_args()
    quantize(args.model, args.out, args.bits, args.group_size)
