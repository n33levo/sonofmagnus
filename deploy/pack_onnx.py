import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def pack(source, output):
    if Path(source).resolve() == Path(output).resolve() or Path(output).exists():
        raise ValueError('Use a distinct new output path')
    model = onnx.load(source)
    nodes, saved = [], 0
    for weight in model.graph.initializer:
        if weight.data_type != TensorProto.FLOAT:
            continue
        values = numpy_helper.to_array(weight)
        with np.errstate(over='ignore', invalid='ignore'):
            half = values.astype(np.float16)
        if values.size <= 64 or not np.array_equal(values.view(np.uint32), half.astype(np.float32).view(np.uint32)):
            continue
        original = weight.name
        packed = original + '/stored_half'
        weight.CopyFrom(numpy_helper.from_array(half, packed))
        nodes.append(helper.make_node('Cast', [packed], [original], to=TensorProto.FLOAT))
        saved += values.nbytes - half.nbytes
    existing = list(model.graph.node)
    model.graph.ClearField('node')
    model.graph.node.extend(nodes + existing)
    onnx.checker.check_model(model)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return {'packed_arrays': len(nodes), 'raw_bytes_saved': saved, 'file_bytes': Path(output).stat().st_size}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lossless half storage for exactly representable float constants')
    parser.add_argument('--in', dest='source', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(pack(args.source, args.out))
