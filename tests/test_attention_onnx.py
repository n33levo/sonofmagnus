import json

import chess
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from deploy.quantize_int8 import quantize_model
from deploy.size_latency_check import check_model
from play.runner import load_runner


def test_classical_int8_runs_on_cpu_and_retains_legal_history(tmp_path):
    rng = np.random.default_rng(0)
    nodes = [helper.make_node('Reshape', ['board', 'shape'], ['flat']),
             helper.make_node('Slice', ['flat', 'start', 'end', 'axis'], ['features']),
             helper.make_node('MatMul', ['features', 'policy_w'], ['policy']),
             helper.make_node('MatMul', ['features', 'value_w'], ['value_logits']),
             helper.make_node('Softmax', ['value_logits'], ['value'], axis=1)]
    arrays = {'shape': np.array([-1, 7168]), 'start': np.array([0]), 'end': np.array([64]), 'axis': np.array([1]),
              'policy_w': rng.normal(size=(64, 1858)).astype(np.float32) * .02,
              'value_w': rng.normal(size=(64, 3)).astype(np.float32) * .02}
    model = helper.make_model(helper.make_graph(nodes, 'test',
        [helper.make_tensor_value_info('board', TensorProto.FLOAT, ['batch', 112, 8, 8])],
        [helper.make_tensor_value_info('policy', TensorProto.FLOAT, ['batch', 1858]),
         helper.make_tensor_value_info('value', TensorProto.FLOAT, ['batch', 3])],
        [numpy_helper.from_array(value, name) for name, value in arrays.items()]),
        opset_imports=[helper.make_opsetid('', 18)])
    model.ir_version = 10
    fp32, int8 = tmp_path / 'fp32.onnx', tmp_path / 'int8.onnx'
    onnx.save(model, fp32)
    calibration = tmp_path / 'calibration.jsonl'
    calibration.write_text(json.dumps({'fen': chess.STARTING_FEN}) + '\n')
    metrics = quantize_model(fp32, int8, calibration_data=calibration)
    assert metrics['size_mb'] < fp32.stat().st_size / 1e6
    assert any(weight.data_type == TensorProto.INT8 for weight in onnx.load(int8).graph.initializer)
    runner = load_runner(int8, device='cpu')
    board = chess.Board()
    for move in ['e2e4', 'e7e5', 'g1f3', 'b8c6']:
        board.push_uci(move)
    before = board.fen(), list(board.move_stack)
    assert runner.select_move(board, deterministic=True) in board.legal_moves
    assert (board.fen(), board.move_stack) == before
    assert runner.forward_calls == 1
    latency = check_model(int8, num_runs=2, num_warmup=0)
    assert latency['positions'] == 1 and latency['neural_forward_calls'] == 2
