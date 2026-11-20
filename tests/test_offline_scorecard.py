from copy import deepcopy
import json

import pytest

from eval.offline_scorecard import run, summarize
from model.artifacts import artifact_info


def evidence():
    games, details = [], []
    rating = 2300
    for i, (score, delta) in enumerate([(1., 2), (.5, None), (0., -5)]):
        game = dict(game_id=str(i), finished_ms=i, rated=True, perf='blitz',
                    own_rating=rating, opponent_rating=2200, opponent='bot' + str(i),
                    rating_diff=delta, own_moves=2, score=score)
        games.append(game)
        rating += delta or 0
        details.append(dict(game_id=str(i), score=score, opponent=game['opponent'],
            opponent_rating=2200, errors100=1, errors300=0,
            recoverable_large_point_drops=1, first_large_drop=dict(ply=2, move='e2e4'),
            features=dict(mean_point_loss=.1, max_point_loss=.3, fraction_100cp=.5, fraction_300cp=0.)))
    account = dict(account='example-bot', title='BOT', utc='2026-10-19T12:00:00Z',
                   blitz=dict(games=2, rating=rating, rd=45))
    quality = dict(game_diagnostics=details, games=3, decisions=6, new_rated_games=2,
                   calibration_training_games=104, calibration_refitted_on_new_games=False,
                   rating_only_squared_error=.2, quality_squared_error=.1, opponent_cluster_95=[.01, .2])
    return games, account, quality


def test_all_results_remain_and_no_elo_is_invented():
    result = summarize(*evidence(), 'example-bot')
    assert result['outcomes'] == dict(win=1, draw=1, loss=1)
    assert result['actual_match_score'] == .5
    assert result['games_without_recorded_delta'] == ['1']
    assert result['recorded_rating_updates'] == 2
    assert result['official_blitz']['rating'] == 2297
    assert result['offline_elo'] is None and result['target_2800_offline_threshold'] is None
    assert result['automatic_rated'] is False


@pytest.mark.parametrize('change', ['missing-game', 'duplicate-game', 'changed-result',
    'rating-gap', 'wrong-account', 'missing-decision', 'nan', 'impossible-count',
    'changed-fraction', 'missing-first-drop', 'refitted'])
def test_corrupted_or_leaking_evidence_is_rejected(change):
    games, account, quality = deepcopy(evidence())
    row = quality['game_diagnostics'][0]
    if change == 'missing-game': quality['game_diagnostics'].pop()
    elif change == 'duplicate-game': quality['game_diagnostics'][1] = row
    elif change == 'changed-result': row['score'] = 0.
    elif change == 'rating-gap': games[-1]['own_rating'] += 1
    elif change == 'wrong-account': account['account'] = 'somebody-else'
    elif change == 'missing-decision': quality['decisions'] -= 1
    elif change == 'nan': row['features']['mean_point_loss'] = float('nan')
    elif change == 'impossible-count': row['errors300'] = 2
    elif change == 'changed-fraction': row['features']['fraction_100cp'] = 0.
    elif change == 'missing-first-drop': row['first_large_drop'] = None
    else: quality['calibration_refitted_on_new_games'] = True
    with pytest.raises(ValueError):
        summarize(games, account, quality, 'example-bot')


def test_file_review_checks_fresh_actions_and_rejects_stale_snapshots(tmp_path):
    games, account, quality = evidence()
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    positions = [dict(game_id=game['game_id'], ply=ply, played_move='e2e4')
                 for game in games for ply in (0, 2)]
    for name, rows in (('games.jsonl', games), ('positions.jsonl', positions)):
        (snapshot / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (snapshot / 'account.json').write_text(json.dumps(account))
    quality['inputs'] = {str(snapshot / name): artifact_info(snapshot / name)
                         for name in ('games.jsonl', 'positions.jsonl')}
    score = tmp_path / 'quality.json'
    score.write_text(json.dumps(quality))
    transfer = tmp_path / 'transfer.json'
    transfer.write_text(json.dumps(dict(completed_models=16, external_games_per_model=300,
        leave_one_checkpoint_out=dict(index_squared_error=.2, mean_only_squared_error=.1))))
    outcome = tmp_path / 'outcome.json'
    outcome.write_text(json.dumps(dict(prospective_checkpoint_cases=1, actual_score=.62,
        checkpoint_ranking_established=False, predictions=dict(recoverable_sum=dict(
            predicted_development_score=.627, beats_mean_baseline=False)))))
    output = tmp_path / 'scorecard.json'
    run(snapshot, score, transfer, outcome, output, 'example-bot')
    assert json.loads(output.read_text())['decisions'] == 6
    with pytest.raises(FileExistsError):
        run(snapshot, score, transfer, outcome, output, 'example-bot')
    # A changed action cannot reuse earlier quality observations, even if the
    # snapshot hashes are copied into the claimed input manifest.
    quality['game_diagnostics'][0]['first_large_drop']['move'] = 'd2d4'
    score.write_text(json.dumps(quality))
    with pytest.raises(ValueError, match='real game'):
        run(snapshot, score, transfer, outcome, tmp_path / 'bad-action.json', 'example-bot')
    assert not (tmp_path / 'bad-action.json').exists()
    (snapshot / 'positions.jsonl').write_text('[]\n')
    with pytest.raises(ValueError, match='Changed quality evidence'):
        run(snapshot, score, transfer, outcome, tmp_path / 'stale.json', 'example-bot')
    assert not (tmp_path / 'stale.json').exists()
