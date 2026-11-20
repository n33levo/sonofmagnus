import numpy as np
import pytest

from eval.history_transfer import compare, ranks


def rows():
    return [dict(model=str(i), history_quality_index=i, external_score=.2 + .1 * i,
                 external_games=300) for i in range(5)]


def test_transfer_distinguishes_a_real_ordering_from_a_reversed_one():
    data = rows()
    result = compare(data)
    assert result['pearson'] == pytest.approx(1.)
    assert result['pair_ordering'] == dict(agree=10, disagree=0, tied=0)
    error = result['leave_one_checkpoint_out']
    assert error['index_squared_error'] < 1e-20 < error['mean_only_squared_error']
    for row in data:
        row['external_score'] = 1 - row['external_score']
    result = compare(data)
    assert result['spearman'] == pytest.approx(-1.)
    assert result['pair_ordering']['disagree'] == 10
    assert result['target_2800_offline_threshold'] is None
    assert not result['offline_elo_conversion_established']


def test_transfer_ranks_ties_by_their_average_position():
    np.testing.assert_array_equal(ranks([3., 1., 1., 2.]), [4., 1.5, 1.5, 3.])


@pytest.mark.parametrize('change', ['duplicate', 'nonfinite', 'unmatched-budget', 'bad-score'])
def test_transfer_rejects_incomparable_observations(change):
    data = rows()
    key, value = dict(duplicate=('model', '0'), nonfinite=('history_quality_index', float('nan')),
        **{'unmatched-budget': ('external_games', 299), 'bad-score': ('external_score', 1.1)})[change]
    data[-1][key] = value
    with pytest.raises(ValueError):
        compare(data)
