import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from deploy.export_mlx import quantize_weights
from model.artifacts import artifact_info


def calibrated_weights(weights, graph, moments, bits, group_size):
    configurations, scales, errors = {}, {}, {}
    names = {node[1][1] for node in graph['nodes'] if node[0] == 'MatMul' and node[1][1] in weights}
    for name in sorted(names):
        original = weights[name]
        if original.ndim != 2 or name not in moments:
            continue
        diagonal = mx.diag(moments[name]).astype(mx.float32)
        if diagonal.shape != (original.shape[0],) or not np.isfinite(np.array(diagonal)).all() or bool(mx.any(diagonal < 0)):
            raise ValueError('Invalid calibration moment')
        diagonal = mx.maximum(diagonal / mx.maximum(diagonal.mean(), 1e-12), 1e-12)
        best = None
        for exponent in (0., .25, .5, .75, 1.):
            scale = diagonal ** (.5 * exponent)
            scale = mx.clip(scale / mx.exp(mx.log(scale).mean()), .01, 100)
            candidate = {name: (original.astype(mx.float32) * scale[:, None]).astype(original.dtype)}
            configuration = quantize_weights(candidate, {name}, bits, group_size)
            if name not in configuration:
                break
            group, width, transpose = configuration[name]
            restored = mx.dequantize(candidate[name + '.packed'], candidate[name + '.scales'],
                                     candidate[name + '.biases'], group_size=group, bits=width).astype(mx.float32)
            inverse = (1 / scale).astype(original.dtype)
            restored = (restored.T if transpose else restored) * inverse.astype(mx.float32)[:, None]
            error = float((mx.square(restored - original.astype(mx.float32)) * diagonal[:, None]).sum())
            if best is None:
                baseline = error
            if best is None or error < best[0]:
                best = error, candidate, configuration[name], inverse, exponent
        if best is None:
            continue
        error, candidate, configuration, inverse, exponent = best
        weights.pop(name)
        weights.update(candidate)
        configurations[name] = configuration
        errors[name] = {'baseline': baseline, 'calibrated': error, 'exponent': exponent}
        if exponent:
            scales[name] = name + '.input_scale'
            weights[scales[name]] = inverse
    configurations.update(quantize_weights(weights, names - configurations.keys(), bits, group_size))
    rewritten = []
    for op, inputs, outputs, attributes in graph['nodes']:
        if op == 'MatMul' and inputs[1] in scales:
            scaled = outputs[0] + '/scaled_input'
            rewritten.append(['Mul', [inputs[0], scales[inputs[1]]], [scaled], {}])
            inputs = [scaled, inputs[1]]
        rewritten.append([op, inputs, outputs, attributes])
    graph['nodes'] = rewritten
    return configurations, errors


def quantize(directory, output, bits=8, group_size=32, calibration=None):
    directory, output = Path(directory), Path(output)
    graph = json.loads((directory / 'graph.json').read_text())
    if graph.get('quantized'):
        raise ValueError('Quantize an unquantized source artifact')
    weights = mx.load(str(directory / 'weights.safetensors'))
    matrices = {node[1][1] for node in graph['nodes'] if node[0] == 'MatMul' and node[1][1] in weights}
    errors = None
    if calibration:
        record = json.loads(Path(calibration).with_suffix('.json').read_text())
        if record['model'] != artifact_info(directory) or record['artifact'] != artifact_info(calibration):
            raise ValueError('Calibration must match the source model and moment artifact')
        graph['quantized'], errors = calibrated_weights(weights, graph, mx.load(str(calibration)), bits, group_size)
    else:
        graph['quantized'] = quantize_weights(weights, matrices, bits, group_size)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'graph.json').write_text(json.dumps(graph, separators=(',', ':')) + '\n')
    mx.save_safetensors(str(output / 'weights.safetensors'), weights)
    metadata = json.loads((directory / 'metadata.json').read_text())
    metadata.update(weight_bits=bits, quantization_group_size=group_size, quantized_matrices=len(graph['quantized']), source_artifact=artifact_info(directory),
                    weights_and_graph_bytes=sum(p.stat().st_size for p in output.iterdir()))
    if calibration:
        metadata.update(calibration=artifact_info(calibration), calibration_metric='Diagonal activation-weighted squared reconstruction error',
                        calibration_errors=errors)
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
    parser.add_argument('--calibration', help='Source-model activation moments from deploy.calibrate_mlx')
    args = parser.parse_args()
    quantize(args.model, args.out, args.bits, args.group_size, args.calibration)
