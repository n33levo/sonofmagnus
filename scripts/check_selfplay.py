import argparse
import json
import math
from pathlib import Path

import chess

from model.artifacts import artifact_info


def check(source, checkpoint):
    source = Path(source)
    metadata = json.loads(source.with_suffix('.json').read_text())
    frozen = artifact_info(source)
    assert frozen == metadata['artifact'] and artifact_info(checkpoint) == metadata['model']
    for name, digest in metadata['runtime_source'].items():
        assert artifact_info(name)['sha256'] == digest, f'Generator source changed: {name}'
    boards, histories, last_block = {}, {}, None
    count = calls = started = finished = 0
    previous_call = None
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            game = int(row['game_id'].removeprefix('selfplay-'))
            assert 0 <= game < metadata['requested_games']
            block = game // metadata['batch_size']
            if block != last_block:
                assert last_block is None or block > last_block
                finished += sum(board.is_game_over(claim_draw=True) for board in boards.values())
                boards, histories, last_block = {}, {}, block
            if game not in boards:
                boards[game], histories[game] = chess.Board(row['root_fen']), []
                started += 1
            board = boards[game]
            assert board.is_valid() and board.root().fen() == row['root_fen']
            assert histories[game] == row['history'] and board.fen() == row['fen']
            assert len(histories[game]) < metadata['max_plies']
            assert 'value' not in row and math.isfinite(row['predicted_value']) and -1 <= row['predicted_value'] <= 1
            assert math.isfinite(row['policy_entropy']) and 0 <= row['policy_entropy'] <= 1
            move = chess.Move.from_uci(row['move'])
            assert move in board.legal_moves and not board.is_game_over(claim_draw=True)
            call = block, len(histories[game])
            if call != previous_call:
                calls += 1
                previous_call = call
            board.push(move)
            histories[game].append(row['move'])
            count += 1
    finished += sum(board.is_game_over(claim_draw=True) for board in boards.values())
    assert count == metadata['positions'] == metadata['neural_input_positions']
    assert calls == metadata['neural_forward_calls']
    assert started == metadata['games_started'] and finished == metadata['games_finished']
    assert artifact_info(source) == frozen
    return dict(positions=count, legal_moves=count, neural_calls=calls, games_started=started,
                games_finished=finished, unfinished_games=started-finished, artifact=frozen,
                model=metadata['model'], reviewer_sha256=artifact_info(__file__)['sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = check(args.source, args.ckpt)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
