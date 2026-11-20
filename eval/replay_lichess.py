"""Replay a searchless model on the exact pre-move histories of past games."""
import argparse
import json
from pathlib import Path

from chessio.records import record_board
from eval.sources import frozen_runtime
from model.artifacts import artifact_info
from play.runner import load_runner


def run(model, positions, output):
    frozen = {str(path): artifact_info(path) for path in (model, positions)}
    temporary = output.with_suffix('.part')
    assert not output.exists() and not temporary.exists()
    runner = load_runner(model)
    count = 0
    with frozen_runtime('eval/replay_lichess.py') as runtime, positions.open() as source, temporary.open('x') as saved:
        for line in source:
            row = json.loads(line)
            board = record_board(row)
            before = runner.forward_calls
            move = runner.select_move(board, deterministic=True)
            if move not in board.legal_moves or runner.forward_calls - before != 1:
                raise ValueError('Expected a legal move and exactly one root forward')
            saved.write(json.dumps(dict(game_id=row['game_id'], ply=row['ply'], fen=row['fen'], move=move.uci())) + '\n')
            count += 1
            if count % 256 == 0:
                saved.flush()
                print(json.dumps(dict(checked=count, forward_calls=runner.forward_calls)), flush=True)
    assert runner.forward_calls == count and all(artifact_info(path) == info for path, info in frozen.items())
    temporary.replace(output)
    report = dict(artifact=artifact_info(output), model=str(model), model_artifact=frozen[str(model)],
        positions=frozen[str(positions)], roots=count, neural_forwards=count, runtime_source=runtime,
        history='Only actual moves before the root; no successor-board inference.', automatic_rated=False,
        scope='Counterfactual root choices on historical trajectories, not counterfactual complete games or Elo.')
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(roots=count, neural_forwards=count)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--positions', type=Path, default=Path('data/lichess-history/positions.jsonl'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.positions, args.out)
