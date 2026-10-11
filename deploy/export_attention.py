import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from model.artifacts import artifact_info


def export(directory, output):
    import mlx.core as mx
    directory, output = Path(directory), Path(output)
    graph = json.loads((directory / 'graph.json').read_text())
    if graph.get('quantized'):
        raise ValueError('Export the unquantized artifact, then quantize ONNX')
    loaded = mx.load(str(directory / 'weights.safetensors'))
    initializers = [numpy_helper.from_array(np.array(value).astype(np.int64 if name in graph['integer_constants'] else np.float32), name)
                    for name, value in loaded.items()]
    nodes = []
    for op, inputs, outputs, attributes in graph['nodes']:
        inputs, attrs = ['board' if name == graph['inputs'][0] else name for name in inputs], dict(attributes)
        key = 'axes' if op == 'ReduceMean' else 'split' if op == 'Split' else None
        if key in attrs:
            name = outputs[0] + '/' + key
            initializers.append(numpy_helper.from_array(np.asarray(attrs.pop(key), dtype=np.int64), name))
            inputs.append(name)
        if op == 'Split' and len(inputs) == 1:
            attrs['num_outputs'] = len(outputs)
        nodes.append(helper.make_node(op, inputs, outputs, name=outputs[0], **attrs))
    nodes += [helper.make_node('Identity', [name], [alias]) for name, alias in zip(graph['outputs'], ('policy', 'value')) if name != alias]
    model = helper.make_model(helper.make_graph(nodes, 'sonofmagnus',
        [helper.make_tensor_value_info('board', TensorProto.FLOAT, ['batch', 112, 8, 8])],
        [helper.make_tensor_value_info('policy', TensorProto.FLOAT, ['batch', 1858]),
         helper.make_tensor_value_info('value', TensorProto.FLOAT, ['batch', 3])], initializers),
        opset_imports=[helper.make_opsetid('', 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    output.with_suffix('.json').write_text(json.dumps({'source_artifact': artifact_info(directory),
        'model': artifact_info(output), 'format': 'FP32 ONNX, classical history'}, indent=2) + '\n')
    return artifact_info(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.model, args.out), indent=2))
