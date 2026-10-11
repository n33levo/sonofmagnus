import numpy as np

from chessio.policy_map import move_to_policy_index
from play.runner import ChessRunner


class PretrainedRunner(ChessRunner):
    def __init__(self, weights, backend='auto'):
        from lczero.backends import Backend, Weights
        weights = Weights(str(weights))
        if weights.input_format() != 1:
            raise ValueError('This adapter supports classical LC0 input; canonical formats need transformed policy indices')
        if backend == 'auto':
            available = Backend.available_backends()
            backend = next(name for name in ('metal', 'cudnn-auto', 'blas', 'eigen') if name in available)
        self.backend = Backend(weights=weights, backend=backend)
        self.temp_opening, self.temp_midgame, self.temp_endgame = .6, .4, .2
        self.last_latency_ms = 0.
        self.forward_calls = 0

    def batch_policy(self, boards):
        from lczero.backends import GameState
        states = [GameState(fen=b.root().fen(), moves=[m.uci() for m in b.move_stack]) for b in boards]
        outputs = self.backend.evaluate(*(state.as_input(self.backend) for state in states))
        self.forward_calls += 1
        result = []
        for board, state, output in zip(boards, states, outputs):
            logits = np.full((8, 8, 73), -np.inf, dtype=np.float32)
            for uci, score in zip(state.moves(), output.p_raw(*state.policy_indices())):
                move = board.parse_uci(uci)
                square, plane = move_to_policy_index(move, board)
                logits[square // 8, square % 8, plane] = score
            if not np.isfinite(output.q()):
                raise ValueError('Non-finite pretrained value output')
            result.append((logits, output.q()))
        return result
