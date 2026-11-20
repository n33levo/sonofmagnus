"""Save a fresh public export without changing the registered benchmark."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

from model.artifacts import artifact_info
from scripts.check_history_rating import reconstruct
from scripts.lichess_history import ACCOUNT, prepare


def refresh(baseline, output):
    if output.exists():
        raise ValueError('Use a new snapshot folder')
    baseline = baseline.resolve()
    frozen = artifact_info(baseline)
    prepare(output)
    request = urllib.request.Request(f'https://lichess.org/api/user/{ACCOUNT}',
        headers={'User-Agent': 'sonofmagnus-local-benchmark'})
    with urllib.request.urlopen(request, timeout=30) as response:
        public = json.load(response)
    if public['id'] != ACCOUNT or public.get('title') != 'BOT':
        raise ValueError('Unexpected Lichess account')
    utc = datetime.now(timezone.utc).isoformat()
    account = dict(utc=utc, account=ACCOUNT, title=public['title'], blitz=public['perfs']['blitz'])
    (output / 'account.json').write_text(json.dumps(account, indent=2) + '\n')
    before = {row['game_id']: row for row in map(json.loads, baseline.open())}
    after = {row['game_id']: row for row in map(json.loads, (output / 'games.jsonl').open())}
    rating = reconstruct(list(after.values()), account)
    (output / 'rating-reconciliation.json').write_text(json.dumps(rating, indent=2) + '\n')
    report = dict(utc=utc, folder=str(output), official=account, old_games=len(before), new_games=len(after),
        added_games=sorted(after.keys() - before.keys()), removed_games=sorted(before.keys() - after.keys()),
        changed_games=sorted(key for key in before.keys() & after.keys() if before[key] != after[key]),
        reconciled=rating['reconciled'],
        positions_identical=artifact_info(output / 'positions.jsonl') == artifact_info(baseline.with_name('positions.jsonl')),
        inputs={str(baseline): frozen, str(Path(__file__).resolve()): artifact_info(__file__)},
        automatic_rated=False,
        scope='Fresh public observation only. New games require a new scoring cohort; the existing benchmark stays frozen.')
    if artifact_info(baseline) != frozen:
        raise ValueError('Registered game snapshot changed')
    (output / 'refresh-review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    refresh(args.baseline, args.out)
