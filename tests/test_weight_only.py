import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

from deploy.weight_only_onnx import quantize
from play.onnx import session_options


def test_weight_only_quantization_uses_output_channels_and_float_activations(tmp_path):
    rng = np.random.default_rng(20261011)
    weights = rng.normal(size=(80, 4)).astype(np.float32) * np.array([.001, .01, .1, 1.], dtype=np.float32)
    model = helper.make_model(helper.make_graph([helper.make_node('MatMul', ['x', 'w'], ['y'])], 'weights',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['batch', 80])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['batch', 4])],
        [numpy_helper.from_array(weights, 'w')]), opset_imports=[helper.make_opsetid('', 18)])
    model.ir_version = 10
    source, output = tmp_path / 'source.onnx', tmp_path / 'quantized.onnx'
    onnx.save(model, source)
    assert quantize(source, output)['quantized_weights'] == 1
    graph = onnx.load(output).graph
    assert [node.op_type for node in graph.node] == ['DequantizeLinear', 'MatMul']
    arrays = {weight.name: numpy_helper.to_array(weight) for weight in graph.initializer}
    restored = arrays['w/int8'].astype(np.float32) * arrays['w/scale'][None, :]
    assert np.all(abs(restored - weights) <= arrays['w/scale'][None, :] * .50001)
    inputs = rng.normal(size=(3, 80)).astype(np.float32)
    options = session_options(output)
    assert options.get_session_config_entry('session.disable_quant_qdq') == '1'
    actual = ort.InferenceSession(str(output), sess_options=options, providers=['CPUExecutionProvider']).run(None, {'x': inputs})[0]
    np.testing.assert_allclose(actual, inputs @ restored, atol=2e-5, rtol=2e-5)


def test_weight_only_grouped_convolution_preserves_the_input_boundary(tmp_path):
    import torch
    rng = np.random.default_rng(20261011)
    identity = np.eye(4, dtype=np.float32).reshape(4, 4, 1, 1)
    kernel = rng.normal(size=(4, 2, 3, 3)).astype(np.float32) * np.array([.001, .01, .1, 1.], dtype=np.float32)[:, None, None, None]
    model = helper.make_model(helper.make_graph([
        helper.make_node('Conv', ['x', 'input'], ['boundary']),
        helper.make_node('Conv', ['boundary', 'body'], ['y'], pads=[1, 1, 1, 1], group=2)], 'convolution',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['batch', 4, 8, 8])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['batch', 4, 8, 8])],
        [numpy_helper.from_array(identity, 'input'), numpy_helper.from_array(kernel, 'body')]),
        opset_imports=[helper.make_opsetid('', 18)])
    model.ir_version = 10
    source, output = tmp_path / 'source.onnx', tmp_path / 'quantized.onnx'
    onnx.save(model, source)
    assert quantize(source, output)['quantized_weights'] == 1
    arrays = {weight.name: numpy_helper.to_array(weight) for weight in onnx.load(output).graph.initializer}
    np.testing.assert_array_equal(arrays['input'], identity)
    restored = arrays['body/int8'].astype(np.float32) * arrays['body/scale'][:, None, None, None]
    assert np.all(abs(restored - kernel) <= arrays['body/scale'][:, None, None, None] * .50001)
    inputs = rng.normal(size=(2, 4, 8, 8)).astype(np.float32)
    actual = ort.InferenceSession(str(output), sess_options=session_options(output), providers=['CPUExecutionProvider']).run(None, {'x': inputs})[0]
    expected = torch.nn.functional.conv2d(torch.from_numpy(inputs), torch.from_numpy(restored), padding=1, groups=2).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
