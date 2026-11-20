import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def quantize(source, output):
    if Path(source).resolve() == Path(output).resolve() or Path(output).exists():
        raise ValueError('Use a distinct new output path')
    model = onnx.load(source)
    weights = {weight.name: weight for weight in model.graph.initializer}
    nodes, added, protected = [], [], set()
    convs = [node for node in model.graph.node if node.op_type == 'Conv']
    if convs:
        protected.add(convs[0].input[1])
    for node in model.graph.node:
        if node.op_type not in ('Conv', 'MatMul') or len(node.input) < 2:
            continue
        name = node.input[1]
        if name in protected or name not in weights or weights[name].data_type != TensorProto.FLOAT:
            continue
        if any(value.startswith(('/policy/', '/value/', '/geometry/')) for value in [node.name, *node.output]):
            continue
        values = numpy_helper.to_array(weights[name])
        if values.ndim not in (2, 4):
            continue
        axis = 0 if node.op_type == 'Conv' else 1
        reduce = tuple(index for index in range(values.ndim) if index != axis)
        scale = np.maximum(np.max(abs(values), axis=reduce) / 127, np.finfo(np.float32).tiny).astype(np.float32)
        shape = [1] * values.ndim
        shape[axis] = values.shape[axis]
        packed = np.clip(np.rint(values / scale.reshape(shape)), -127, 127).astype(np.int8)
        weights[name].CopyFrom(numpy_helper.from_array(packed, name + '/int8'))
        added.extend([numpy_helper.from_array(scale, name + '/scale'),
                      numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), name + '/zero')])
        nodes.append(helper.make_node('DequantizeLinear', [name + '/int8', name + '/scale', name + '/zero'], [name], axis=axis))
        protected.add(name)
    original = list(model.graph.node)
    model.graph.ClearField('node')
    model.graph.node.extend(nodes + original)
    model.graph.initializer.extend(added)
    metadata = model.metadata_props.add()
    metadata.key, metadata.value = 'sonofmagnus.float_activations', '1'
    onnx.checker.check_model(model)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return {'quantized_weights': len(nodes), 'file_bytes': Path(output).stat().st_size,
            'activation_precision': 'float32', 'weight_precision': 'int8, per-output-channel; float boundaries'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Weight-only INT8 with float activations')
    parser.add_argument('--in', dest='source', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(quantize(args.source, args.out))
