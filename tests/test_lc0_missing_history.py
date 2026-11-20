import json
from pathlib import Path

import chess
import numpy as np
import pytest

from chessio.lc0 import encode_history, policy_index


FIXTURE = json.loads((Path(__file__).parent / 'fixtures/lc0_missing_ep_history.json').read_text())


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda case: case['name'])
def test_missing_history_preserves_native_ep_reconstruction(case):
    board = chess.Board(case['root_fen'])
    assert board.is_valid() and board.ep_square is not None and not board.has_legal_en_passant()
    before = board.fen(en_passant='fen'), tuple(board.move_stack)
    expected = np.array([[(int(mask) >> bit & 1) * value for bit in range(64)]
        for mask, value in case['native']['planes']], np.float32).reshape(112, 8, 8)
    np.testing.assert_array_equal(encode_history(board), expected)
    assert [policy_index(board.parse_uci(move), board) for move in case['moves']] == case['native']['indices']
    assert before == (board.fen(en_passant='fen'), tuple(board.move_stack))
