import argparse
import json
from pathlib import Path

from chessio.records import holdout_positions, record_board
from model.artifacts import artifact_info
from scripts.review_lichess import retained_aborts, review


def export(run, output, exclude=()):
    run, output = Path(run), Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Refusing to replace an existing replay')
    files = [run / name for name in ('manifest.json', 'server_games.jsonl', 'moves.jsonl')]
    frozen = {str(path): artifact_info(path) for path in files}
    manifest = json.loads(files[0].read_text())
    runtime = run / manifest['runtime_directory'] if 'runtime_directory' in manifest else Path(__file__).resolve().parents[1]
    if any(artifact_info(runtime / name)['sha256'] != digest for name, digest in manifest['runtime_source'].items()):
        raise ValueError('Frozen serving source changed')
    if artifact_info(run / Path(manifest['source_checkpoint']).name) != manifest['model']:
        raise ValueError('Frozen serving model changed')
    games = [json.loads(line) for line in files[1].read_text().splitlines()]
    aborts = retained_aborts(run / 'games', {item['id'] for item in games})
    records = [json.loads(line) for line in files[2].read_text().splitlines()]
    played = []
    audit = review([*games, *aborts], records, manifest['account'], played_records=played)
    if audit['unmatched_decisions'] or len(played) != audit['neural_moves_verified']:
        raise ValueError('Unmatched or incomplete live decisions')
    blocked, rows = holdout_positions(exclude), []
    for game_id, row in played:
        board = record_board(row)
        if ' '.join(board.fen().split()[:4]) not in blocked and not board.is_game_over(claim_draw=True):
            rows.append({**row, 'game_id': game_id, 'source': 'verified-rated-blitz-replay'})
    if not rows or any(artifact_info(path) != identity for path, identity in frozen.items()):
        raise ValueError('No eligible rows or changed live files')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.writelines(json.dumps(row, separators=(',', ':')) + '\n' for row in rows)
    result = dict(positions=len(rows), accepted_decisions=len(played), excluded_decisions=len(played) - len(rows),
                  games=audit['games'], cancelled_decisions=len(audit['cancelled_decisions']),
                  source=frozen, model=manifest['model'], artifact=artifact_info(output),
                  excluded_inputs=list(exclude), producer=artifact_info(__file__), reviewer=artifact_info(Path(__file__).with_name('review_lichess.py')),
                  retained_aborted_pgns={item['id']: item['source_artifact'] for item in aborts},
                  labels='Recorded engine decisions; require offline teacher labels before supervised training')
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--exclude-data', required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.out, args.exclude_data.split(',')), indent=2))
