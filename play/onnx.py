import numpy as np
import onnxruntime as ort

from chessio.encode import encode_board_batch
from play.runner import ChessRunner


class OnnxRunner(ChessRunner):
    def __init__(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        specification = self.session.get_inputs()[0]
        if specification.shape[1:] != [18, 8, 8]:
            raise ValueError('ONNX runner expects an 18-plane board encoder')
        self.input_name = specification.name
        self.dtype = np.float16 if specification.type == 'tensor(float16)' else np.float32
        self.temp_opening, self.temp_midgame, self.temp_endgame = .6, .4, .2
        self.last_latency_ms = 0.
        self.forward_calls = 0

    def batch_policy(self, boards):
        tensor = encode_board_batch(boards).astype(self.dtype, copy=False)
        policy, value = self.session.run(['policy', 'value'], {self.input_name: tensor})
        self.forward_calls += 1
        if not np.isfinite(value).all():
            raise ValueError('Non-finite ONNX value output')
        return list(zip(policy.astype(np.float32), value.reshape(-1).tolist()))
