import numpy as np
import onnxruntime as ort

from chessio.encode import encode_board_batch
from chessio.lc0 import encode_history, policy_index
from chessio.policy_map import move_to_policy_index
from play.runner import ChessRunner


class OnnxRunner(ChessRunner):
    def __init__(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        specification = self.session.get_inputs()[0]
        if specification.shape[1:] not in ([18, 8, 8], [112, 8, 8]):
            raise ValueError('Expected an 18-plane native or 112-plane classical attention input')
        self.classical = specification.shape[1] == 112
        self.input_name = specification.name
        self.dtype = np.float16 if specification.type == 'tensor(float16)' else np.float32
        self.temp_opening, self.temp_midgame, self.temp_endgame = .6, .4, .2
        self.last_latency_ms = 0.
        self.forward_calls = 0

    def batch_policy(self, boards):
        tensor = (np.stack([encode_history(board) for board in boards]) if self.classical else encode_board_batch(boards)).astype(self.dtype, copy=False)
        policy, value = self.session.run(['policy', 'value'], {self.input_name: tensor})
        self.forward_calls += 1
        if not np.isfinite(value).all():
            raise ValueError('Non-finite ONNX value output')
        if self.classical:
            if policy.shape != (len(boards), 1858) or value.shape != (len(boards), 3) or not np.isfinite(policy).all():
                raise ValueError('Invalid classical policy or WDL outputs')
            if (value < 0).any() or (value > 1).any() or not np.allclose(value.sum(-1), 1, atol=.002):
                raise ValueError('Invalid ONNX WDL probabilities')
            result = []
            for board, scores, wdl in zip(boards, policy, value):
                logits = np.full((8, 8, 73), -np.inf, np.float32)
                for move in board.legal_moves:
                    square, plane = move_to_policy_index(move, board)
                    logits[square // 8, square % 8, plane] = scores[policy_index(move, board)]
                result.append((logits, float(wdl[0] - wdl[2])))
            return result
        return list(zip(policy.astype(np.float32), value.reshape(-1).tolist()))
