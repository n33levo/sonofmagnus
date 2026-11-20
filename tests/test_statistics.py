import math

import pytest

from eval.statistics import binomial_interval


def test_exact_intervals_include_extreme_binomial_cases():
    assert binomial_interval(0, 10) == pytest.approx([0., 1 - .025 ** .1])
    assert binomial_interval(10, 10) == pytest.approx([.025 ** .1, 1.])
    assert binomial_interval(1, 2) == pytest.approx([1 - math.sqrt(.975), math.sqrt(.975)])
    lower, upper = binomial_interval(5, 10)
    assert lower == pytest.approx(1 - upper)
    wider = binomial_interval(5, 10, .01)
    assert wider[0] < lower < upper < wider[1]
