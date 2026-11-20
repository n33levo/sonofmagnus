import argparse
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from chessio.encode import encode_board
from chessio.lc0 import encode_history
from chessio.records import record_board


class BoardCalibration(CalibrationDataReader):
    def __init__(self, path, limit=1000, classical=False):
        if limit <= 0:
            raise ValueError('Calibration size must be positive')
        lines, seen, rng = [], 0, random.Random(0)
        with open(path) as stream:
            for line in stream:
                if not line.strip():
                    continue
                seen += 1
                index = len(lines) if len(lines) < limit else rng.randrange(seen)
                if index < limit:
                    if index == len(lines):
                        lines.append(line)
                    else:
                        lines[index] = line
        encoder = encode_history if classical else encode_board
        self.positions = [record_board(json.loads(line)) for line in lines]
        self.boards = [encoder(board)[None] for board in self.positions]
        if not self.boards:
            raise ValueError('Empty calibration set')
        self.rewind()

    def get_next(self):
        board = next(self.iterator, None)
        return None if board is None else {'board': board}

    def rewind(self):
        self.iterator = iter(self.boards)


def quantize_model(input_path, output_path, weight_type='int8', calibration_data=None, keep_input_and_heads=False, keep_convs=()):
    if calibration_data is None:
        raise ValueError('INT8 conversion requires --calibration-data for validation')
    model = onnx.load(input_path)
    if model.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError('Quantize an FP32 export, not FP16')
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    classical = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value == 112
    cnn = any(node.op_type == 'Conv' for node in model.graph.node)
    if keep_input_and_heads and not (classical and cnn):
        raise ValueError('Keeping these boundaries in FP32 requires a classical LC0 CNN')
    calibration = BoardCalibration(calibration_data, classical=classical)
    with tempfile.TemporaryDirectory() as directory:
        prepared = str(Path(directory) / 'prepared.onnx')
        quant_pre_process(input_path, prepared)
        excluded = []
        if keep_input_and_heads:
            optimized = onnx.load(prepared)
            first_conv = next(node.name for node in optimized.graph.node if node.op_type == 'Conv')
            excluded = [node.name for node in optimized.graph.node
                        if any(name.startswith(('/inputconv/', '/geometry/', '/policy/', '/value/')) for name in [node.name, *node.output])
                        or (node.op_type == 'Conv' and node.name == first_conv)]
        if keep_convs:
            convs = [node.name for node in onnx.load(prepared).graph.node if node.op_type == 'Conv']
            if any(index < 0 or index >= len(convs) for index in keep_convs):
                raise ValueError('Protected convolution index out of range')
            excluded.extend(convs[index] for index in keep_convs)
        if classical and not cnn:
            # Recompute activation ranges per position: history includes the unbounded rule-50 clock.
            quantize_dynamic(prepared, str(output), weight_type=QuantType.QInt8 if weight_type == 'int8' else QuantType.QUInt8,
                             per_channel=True, op_types_to_quantize=['MatMul'])
        else:
            quantize_static(prepared, str(output), calibration, quant_format=QuantFormat.QDQ,
                            activation_type=QuantType.QUInt8,
                            weight_type=QuantType.QInt8 if weight_type == 'int8' else QuantType.QUInt8,
                            per_channel=True, op_types_to_quantize=['Conv', 'MatMul', 'Gemm'], nodes_to_exclude=excluded)
    model = onnx.load(output)
    public = {value.name for value in [*model.graph.input, *model.graph.output]}
    names = {name for node in model.graph.node for name in [*node.input, *node.output] if name and name not in public}
    names.update(value.name for value in model.graph.initializer if value.name not in public)
    compact = {name: f'v{i}' for i, name in enumerate(sorted(names))}
    for node in model.graph.node:
        node.name = ''
        for values in (node.input, node.output):
            for i, name in enumerate(values):
                values[i] = compact.get(name, name)
    for value in model.graph.initializer:
        value.name = compact.get(value.name, value.name)
    model.graph.ClearField('value_info')
    onnx.checker.check_model(model)
    onnx.save(model, output)
    baseline = ort.InferenceSession(str(input_path), providers=['CPUExecutionProvider'])
    quantized = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    errors = []
    for board in calibration.boards[:20]:
        reference = baseline.run(None, {'board': board})
        actual = quantized.run(None, {'board': board})
        if not all(np.isfinite(value).all() for value in actual):
            raise FloatingPointError('Quantized model produced non-finite output')
        errors.append(float(np.abs(reference[0] - actual[0]).max()))
    print(f'INT8: {output.stat().st_size / 1e6:.2f} MB; max calibration logit error {max(errors):.4f}')
    return {'size_mb': output.stat().st_size / 1e6, 'max_calibration_logit_error': max(errors)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', '--model', dest='input', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--weight-type', choices=['int8', 'uint8'], default='int8')
    parser.add_argument('--calibration-data', required=True)
    parser.add_argument('--keep-input-and-heads', action='store_true', help='Spend extra bytes on FP32 input convolution and heads')
    parser.add_argument('--keep-convs', default='', help='Comma-separated convolution indices to preserve in float')
    args = parser.parse_args()
    quantize_model(args.input, args.out, args.weight_type, args.calibration_data, args.keep_input_and_heads,
                   tuple(map(int, args.keep_convs.split(','))) if args.keep_convs else ())


if __name__ == '__main__':
    main()
