import argparse
from pathlib import Path

import chess
import numpy as np
import onnx
import onnxruntime as ort
import torch

from chessio.encode import encode_board_batch
from model import load_model


def export_to_onnx(checkpoint_path, output_path, config='main', fp16=False, opset_version=18):
    if opset_version < 18:
        raise ValueError('GroupNorm export requires ONNX opset 18 or later')
    model = load_model(checkpoint_path, config)
    if fp16:
        model.half()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if fp16 else torch.float32
    boards = [chess.Board(), chess.Board('7k/8/8/3pP3/8/8/8/K7 w - d6 0 1')]
    sample = torch.from_numpy(encode_board_batch(boards)).to(dtype)
    torch.onnx.export(model, (sample,), str(output), opset_version=opset_version, dynamo=True, external_data=False,
                      input_names=['board'], output_names=['policy', 'value'],
                      dynamic_shapes={'x': {0: torch.export.Dim('batch', min=1)}})
    onnx.checker.check_model(onnx.load(output))
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    with torch.inference_mode():
        expected = model(sample)
    actual = session.run(None, {'board': sample.numpy()})
    tolerance = 1e-2 if fp16 else 1e-4
    for reference, value in zip(expected, actual):
        np.testing.assert_allclose(value, reference.numpy(), rtol=tolerance, atol=tolerance)
    print(f'Exported and verified {output}: {output.stat().st_size / 1e6:.2f} MB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', choices=['main', 'mini'], default='main')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--opset', type=int, default=18)
    args = parser.parse_args()
    export_to_onnx(args.ckpt, args.out, args.config, args.fp16, args.opset)


if __name__ == '__main__':
    main()
