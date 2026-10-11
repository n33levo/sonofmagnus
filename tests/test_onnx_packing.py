import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
import pytest

from deploy.pack_onnx import pack


def test_half_storage_is_bit_exact_and_keeps_nonrepresentable_constants_float(tmp_path):
    rng = np.random.default_rng(20261011)
    exact = rng.normal(0, .01, (80, 3)).astype(np.float16).astype(np.float32)
    exact[0] = [-0., 0., 2. ** -24]
    approximate = np.full((80, 3), .1, dtype=np.float32)
    model = helper.make_model(helper.make_graph([
        helper.make_node('MatMul', ['x', 'exact'], ['a']),
        helper.make_node('MatMul', ['x', 'approximate'], ['b']),
        helper.make_node('Add', ['a', 'b'], ['y'])], 'packing',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['batch', 80])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['batch', 3])],
        [numpy_helper.from_array(exact, 'exact'), numpy_helper.from_array(approximate, 'approximate')]),
        opset_imports=[helper.make_opsetid('', 18)])
    model.ir_version = 10
    source, output = tmp_path / 'source.onnx', tmp_path / 'packed.onnx'
    onnx.save(model, source)
    result = pack(source, output)
    assert result['packed_arrays'] == 1 and result['raw_bytes_saved'] == exact.nbytes // 2
    weights = {weight.name: weight for weight in onnx.load(output).graph.initializer}
    assert weights['approximate'].data_type == TensorProto.FLOAT
    restored = numpy_helper.to_array(weights['exact/stored_half']).astype(np.float32)
    np.testing.assert_array_equal(restored.view(np.uint32), exact.view(np.uint32))
    inputs = rng.normal(size=(4, 80)).astype(np.float32)
    baseline = ort.InferenceSession(str(source), providers=['CPUExecutionProvider']).run(None, {'x': inputs})[0]
    packed = ort.InferenceSession(str(output), providers=['CPUExecutionProvider']).run(None, {'x': inputs})[0]
    np.testing.assert_array_equal(packed.view(np.uint32), baseline.view(np.uint32))
    with pytest.raises(ValueError, match='new output'):
        pack(source, output)
