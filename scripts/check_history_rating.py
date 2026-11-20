"""Reconcile recorded blitz updates with the authenticated account snapshot."""
import argparse
import json
from pathlib import Path

from model.artifacts import artifact_info


def reconstruct(games, account):
    if account['account'] != 'n33levo' or account['title'] != 'BOT':
        raise ValueError('Expected the n33levo BOT account snapshot')
    ordered = sorted(games, key=lambda row: (row['finished_ms'], row['game_id']))
    known = [row for row in ordered if row['rating_diff'] is not None]
    if not known or len({row['game_id'] for row in ordered}) != len(ordered):
        raise ValueError('Require distinct games and recorded rating updates')
    trace, previous = [], None
    for row in known:
        if (not row['rated'] or row['perf'] != 'blitz' or
                type(row['own_rating']) is not int or type(row['rating_diff']) is not int):
            raise ValueError('Expected integer rated-blitz observations')
        if previous is not None and row['own_rating'] != previous:
            raise ValueError('Recorded rating transition has a gap: ' + row['game_id'])
        previous = row['own_rating'] + row['rating_diff']
        trace.append(dict(game_id=row['game_id'], before=row['own_rating'],
            change=row['rating_diff'], after=previous))
    if len(known) != account['blitz']['games'] or previous != account['blitz']['rating']:
        raise ValueError('Game history and account snapshot do not reconcile')
    return dict(games=len(ordered), recorded_updates=len(known), exact_transitions=len(known) - 1,
        initial_rating=known[0]['own_rating'], reconstructed_rating=previous,
        official_blitz=account['blitz'], account_checked_utc=account['utc'],
        missing_deltas=[row['game_id'] for row in ordered if row['rating_diff'] is None],
        trace=trace, reconciled=True, automatic_rated=False, target_achieved=False,
        scope='Exact reconstruction from recorded server updates, not an offline rating prediction for a new model. '
              'The account history used multiple models. Starting ratings do not establish strength milestones; '
              'historical RD and provisional flags are unavailable here.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=Path, default=Path('data/lichess-history/games.jsonl'))
    parser.add_argument('--account', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.games, args.account, Path(__file__)]
    frozen = {str(path.resolve()): artifact_info(path) for path in paths}
    result = reconstruct(list(map(json.loads, args.games.open())), json.loads(args.account.read_text()))
    assert all(artifact_info(path) == info for path, info in frozen.items())
    args.out.write_text(json.dumps(dict(result, inputs=frozen), indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'trace'}))
