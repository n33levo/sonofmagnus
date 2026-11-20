"""Summarize verified Lichess history and the limits of its offline score."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from model.artifacts import artifact_info
from scripts.check_history_rating import reconstruct


def summarize(games, account, quality, expected_account):
    rating = reconstruct(games, account, expected_account)
    records = {row['game_id']: row for row in games}
    details = quality['game_diagnostics']
    if (len(details) != len(records) or {row['game_id'] for row in details} != records.keys()
            or quality['games'] != len(records)
            or quality['decisions'] != sum(row['own_moves'] for row in games)
            or quality['calibration_refitted_on_new_games'] is not False):
        raise ValueError('The quality score must cover every completed game and decision')
    for row in details:
        game = records[row['game_id']]
        if any(row[key] != game[key] for key in ('score', 'opponent', 'opponent_rating')):
            raise ValueError('Quality observations disagree with the server game')
        features = row['features']
        if (game['score'] not in (0., .5, 1.) or game['own_moves'] <= 0
                or any(not math.isfinite(value) or not 0 <= value <= 1 for value in features.values())
                or features['mean_point_loss'] > features['max_point_loss']
                or features['fraction_300cp'] > features['fraction_100cp']):
            raise ValueError('Invalid outcome or move-quality features')
        counts = [row[key] for key in ('errors100', 'errors300', 'recoverable_large_point_drops')]
        if (any(type(value) is not int or not 0 <= value <= game['own_moves'] for value in counts)
                or row['errors300'] > row['errors100']
                or any(not math.isclose(row['errors' + str(cp)] / game['own_moves'],
                                        features['fraction_' + str(cp) + 'cp'], abs_tol=1e-12)
                       for cp in (100, 300))
                or bool(row['first_large_drop']) != bool(row['recoverable_large_point_drops'])):
            raise ValueError('Invalid error counts or first-drop evidence')
    groups = {}
    for name, value in (('wins', 1.), ('draws', .5), ('losses', 0.)):
        rows = [row for row in details if row['score'] == value]
        groups[name] = dict(games=len(rows), games_with_recoverable_large_drop=sum(
            row['recoverable_large_point_drops'] > 0 for row in rows),
            recoverable_large_drops=sum(row['recoverable_large_point_drops'] for row in rows),
            mean_point_loss_per_game=(sum(row['features']['mean_point_loss'] for row in rows)
                                      / len(rows) if rows else None))
    cohort = [row for row in games if row['rating_diff'] is not None]
    return dict(account=account['account'], account_checked_utc=account['utc'],
        official_blitz=account['blitz'], rating_reconciled=rating['reconciled'],
        games=len(games), decisions=quality['decisions'],
        outcomes=dict(Counter('win' if row['score'] == 1 else 'loss' if row['score'] == 0 else 'draw'
                              for row in games)),
        actual_match_score=sum(row['score'] for row in games) / len(games),
        recorded_rating_updates=len(cohort), games_without_recorded_delta=rating['missing_deltas'],
        rated_cohort_match_score=sum(row['score'] for row in cohort) / len(cohort),
        move_quality_definition=dict(point_loss='Independent best expected points minus played-move expected points, clipped at zero.',
            large_drop_threshold=.3,
            recoverable='Independent best score is at least -300cp or a positive mate score.'),
        move_quality_by_outcome=groups,
        later_game_quality_check={key: quality[key] for key in (
            'new_rated_games', 'calibration_training_games', 'calibration_refitted_on_new_games',
            'rating_only_squared_error', 'quality_squared_error', 'opponent_cluster_95')},
        offline_elo=None, target_2800_offline_threshold=None, automatic_rated=False,
        scope='All completed games retained, including games without a recorded rating delta. '
              'Move-quality scores use completed game histories and track mistakes; they are not pregame forecasts. '
              'The account used multiple models, so this is not a rating for one checkpoint.')


def run(snapshot, history_score, transfer, forecast_outcome, output, expected_account):
    paths = [snapshot / name for name in ('games.jsonl', 'positions.jsonl', 'account.json')]
    paths += [history_score, transfer, forecast_outcome, Path(__file__),
              Path(__file__).parents[1] / 'scripts/check_history_rating.py']
    frozen = {str(path.resolve()): artifact_info(path) for path in paths}
    quality = json.loads(history_score.read_text())
    for name, identity in quality['inputs'].items():
        if artifact_info(name) != identity:
            raise ValueError('Changed quality evidence: ' + name)
        frozen[name] = identity
    for name in ('games.jsonl', 'positions.jsonl'):
        if not any(Path(path).name == name and identity == artifact_info(snapshot / name)
                   for path, identity in quality['inputs'].items()):
            raise ValueError('Fresh games changed; score every new decision before summarizing')
    positions = list(map(json.loads, (snapshot / 'positions.jsonl').open()))
    roots = {(row['game_id'], row['ply']): row for row in positions}
    if len(roots) != len(positions) or len(roots) != quality['decisions']:
        raise ValueError('Incomplete or duplicated decision histories')
    for row in quality['game_diagnostics']:
        first = row['first_large_drop']
        if first is not None and roots.get((row['game_id'], first['ply']), {}).get('played_move') != first['move']:
            raise ValueError('The first-drop action does not match the real game')
    result = summarize(list(map(json.loads, (snapshot / 'games.jsonl').open())),
                       json.loads((snapshot / 'account.json').read_text()), quality, expected_account)
    transfer_review = json.loads(transfer.read_text())
    outcome = json.loads(forecast_outcome.read_text())
    result['checkpoint_transfer'] = dict(models=transfer_review['completed_models'],
        games_per_model=transfer_review['external_games_per_model'],
        leave_one_checkpoint_out={key: value for key, value in
            transfer_review['leave_one_checkpoint_out'].items() if key != 'details'},
        prospective_cases=outcome['prospective_checkpoint_cases'],
        latest_actual_engine_score=outcome['actual_score'],
        latest_primary_forecast=outcome['predictions']['recoverable_sum']['predicted_development_score'],
        latest_forecast_beats_mean_baseline=outcome['predictions']['recoverable_sum']['beats_mean_baseline'],
        reliable_checkpoint_ranking_established=outcome['checkpoint_ranking_established'])
    if any(artifact_info(path) != identity for path, identity in frozen.items()):
        raise ValueError('Evidence changed during scoring')
    result.update(utc=datetime.now(timezone.utc).isoformat(), inputs=frozen)
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with output.open('x') as stream:
        stream.write(payload)
    print(json.dumps({key: value for key, value in result.items() if key not in ('inputs', 'scope')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'history-score', 'transfer', 'forecast-outcome', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--account', required=True)
    args = parser.parse_args()
    run(args.snapshot, args.history_score, args.transfer, args.forecast_outcome, args.out, args.account)
