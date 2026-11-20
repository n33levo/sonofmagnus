import json
from unittest.mock import patch

import chess
import numpy as np
import pytest

from chessio.policy_map import move_to_policy_index
from model.artifacts import artifact_info
from scripts.check_selfplay import check
from train.selfplay import play_selfplay_game, run_selfplay


class ScriptPolicy:
    def __init__(self):
        self.games, self.widths, self.forward_calls = {}, [], 0

    def get_temperature(self, move_number):
        return 0

    def batch_policy(self, boards):
        self.forward_calls += 1
        self.widths.append(len(boards))
        outputs = []
        for board in boards:
            if id(board) not in self.games:
                self.games[id(board)] = (board, len(self.games))
            game = self.games[id(board)][1]
            sequence = ('e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5'
                        if game == 1 else 'f2f3 e7e5 g2g4 d8h4').split()
            move = chess.Move.from_uci(sequence[len(board.move_stack)])
            assert move in board.legal_moves
            square, plane = move_to_policy_index(move, board)
            logits = np.full((8, 8, 73), -1000.)
            logits.reshape(64, 73)[square, plane] = 1000
            outputs.append((logits, 0.))
        return outputs


def test_refill_reuses_finished_slots_without_waiting_for_long_games(tmp_path):
    checkpoint, output = tmp_path / 'model.pt', tmp_path / 'games.jsonl'
    checkpoint.write_bytes(b'frozen')
    runner = ScriptPolicy()
    with patch('train.selfplay.load_runner', return_value=runner):
        run_selfplay(checkpoint, output, num_games=4, max_moves=12, batch_size=2)
    result = check(output, checkpoint)
    assert result['positions'] == 24 and result['neural_calls'] == 12
    assert result['games_started'] == 4 and result['games_finished'] == 3
    assert runner.widths == [2] * 12
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[8]['game_id'] == 'selfplay-1'
    assert rows[9]['game_id'] == 'selfplay-2' and rows[9]['batch_id'] == 4
    rows[9]['batch_id'] = 3
    output.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    metadata = json.loads(output.with_suffix('.json').read_text())
    metadata['artifact'] = artifact_info(output)
    output.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(AssertionError):
        check(output, checkpoint)


def test_refill_rejects_model_mutation_before_publishing(tmp_path):
    checkpoint, output = tmp_path / 'model.pt', tmp_path / 'games.jsonl'
    checkpoint.write_bytes(b'frozen')
    runner = ScriptPolicy()
    original = runner.batch_policy

    def changing(boards):
        checkpoint.write_bytes(b'changed')
        return original(boards)

    runner.batch_policy = changing
    with patch('train.selfplay.load_runner', return_value=runner), pytest.raises(ValueError, match='changed model'):
        run_selfplay(checkpoint, output, num_games=1, positions=1)
    assert not output.exists() and not output.with_suffix('.json').exists()


@pytest.mark.parametrize('avoid, length', [(True, 12), (False, 8)])
def test_selfplay_can_avoid_a_prospective_draw_or_play_the_third_occurrence(tmp_path, avoid, length):
    moves = ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1']
    moves += ['f6e4', 'g2g4', 'e7e5', 'f2f3', 'd8h4'] if avoid else ['f6g8']
    class Runner:
        forward_calls = 0
        def get_temperature(self, move_number):
            return 0
        def select_move(self, board):
            return board.parse_uci(moves[len(board.move_stack)])
        def batch_policy(self, boards):
            self.forward_calls += 1
            outputs = []
            for board in boards:
                logits = np.full((8, 8, 73), -1000.)
                square, plane = move_to_policy_index(self.select_move(board), board)
                logits.reshape(64, 73)[square, plane] = 1000
                outputs.append((logits, 0.))
            return outputs
    checkpoint, output = tmp_path / 'model.pt', tmp_path / 'games.jsonl'
    checkpoint.write_bytes(b'frozen')
    assert len(play_selfplay_game(Runner())) == length
    runner = Runner()
    with patch('train.selfplay.load_runner', return_value=runner):
        run_selfplay(checkpoint, output, num_games=1, batch_size=1)
    result = check(output, checkpoint)
    assert result['positions'] == result['neural_calls'] == length
    assert result['games_finished'] == 1
    assert json.loads(output.with_suffix('.json').read_text())['draw_claim_protocol'] == 'present_position'
    if not avoid:
        # Historical generators stopped before playing the prospective repetition.
        output.write_text(''.join(output.read_text().splitlines(keepends=True)[:7]))
        metadata = json.loads(output.with_suffix('.json').read_text())
        metadata.update(positions=7, neural_input_positions=7, neural_forward_calls=7, artifact=artifact_info(output))
        output.with_suffix('.json').write_text(json.dumps(metadata))
        with pytest.raises(AssertionError):
            check(output, checkpoint)
        del metadata['draw_claim_protocol']
        output.with_suffix('.json').write_text(json.dumps(metadata))
        assert check(output, checkpoint)['games_finished'] == 1
