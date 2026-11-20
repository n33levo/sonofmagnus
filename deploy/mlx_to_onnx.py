import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from model.artifacts import artifact_info


def export(directory, output):
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise ValueError('Use a new output path')
    frozen = artifact_info(directory)
    graph = json.loads((directory / 'graph.json').read_text())
    if len(graph['inputs']) != 1 or len(graph['outputs']) != 2:
        raise ValueError('Export a classical LC0 graph with policy and WDL outputs')
    weights = mx.load(str(directory / 'weights.safetensors'))
    initializers, nodes, packed_names = [], [], set()
    for name, configuration in graph.get('quantized', {}).items():
        group, bits = configuration[:2]
        transpose = configuration[2] if len(configuration) > 2 else True
        if bits != 8:
            raise ValueError('CPU blockwise export supports eight-bit weights')
        packed = np.array(weights[name + '.packed'])
        codes = ((packed[..., None] >> np.arange(0, 32, 8, dtype=np.uint32)) & 255).astype(np.uint8)
        rows, columns = packed.shape[0], packed.shape[1] * 4
        prefix = name + '/q8'
        arrays = {'codes': codes.reshape(rows, -1, group),
                  'scales': np.array(weights[name + '.scales']).reshape(rows, -1, 1),
                  'biases': np.array(weights[name + '.biases']).reshape(rows, -1, 1),
                  'shape': np.array([rows, columns], dtype=np.int64)}
        initializers.extend(numpy_helper.from_array(value, prefix + '/' + key) for key, value in arrays.items())
        nodes.extend(helper.make_node('Cast', [prefix + '/' + key], [prefix + '/' + key + '_float'], to=TensorProto.FLOAT)
                     for key in ('codes', 'scales', 'biases'))
        nodes.extend([helper.make_node('Mul', [prefix + '/codes_float', prefix + '/scales_float'], [prefix + '/scaled']),
                      helper.make_node('Add', [prefix + '/scaled', prefix + '/biases_float'], [prefix + '/decoded']),
                      helper.make_node('Reshape', [prefix + '/decoded', prefix + '/shape'], [prefix + '/matrix' if transpose else name])])
        if transpose:
            nodes.append(helper.make_node('Transpose', [prefix + '/matrix'], [name], perm=[1, 0]))
        packed_names.update(name + suffix for suffix in ('.packed', '.scales', '.biases'))
    for name, weight in weights.items():
        if name in packed_names:
            continue
        value = np.array(weight)
        dtype = np.int64 if name in graph['integer_constants'] else np.float32
        initializers.append(numpy_helper.from_array(value.astype(dtype), name))
    nodes.extend(helper.make_node(op, inputs, outputs, **attributes)
                 for op, inputs, outputs, attributes in graph['nodes'])
    nodes.extend(helper.make_node('Identity', [source], [target])
                 for source, target in zip(graph['outputs'], ('policy', 'value')) if source != target)
    definition = helper.make_graph(nodes, 'sonofmagnus',
        [helper.make_tensor_value_info(graph['inputs'][0], TensorProto.FLOAT, ['batch', 112, 8, 8])],
        [helper.make_tensor_value_info('policy', TensorProto.FLOAT, ['batch', 1858]),
         helper.make_tensor_value_info('value', TensorProto.FLOAT, ['batch', 3])], initializers)
    opset = 18 if any('num_outputs' in node[3] for node in graph['nodes']) else 17
    model = helper.make_model(definition, opset_imports=[helper.make_opsetid('', opset)], ir_version=9)
    helper.set_model_props(model, {'sonofmagnus.source_sha256': frozen['sha256'],
                                 'sonofmagnus.activation_precision': 'float32',
                                 'sonofmagnus.float_activations': '1'})
    onnx.checker.check_model(model, full_check=True)
    if artifact_info(directory) != frozen:
        raise ValueError('Source model changed during export')
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return artifact_info(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export classical LC0 MLX graphs, including blockwise INT8, for FP32 CPU inference')
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.model, args.out), indent=2))
