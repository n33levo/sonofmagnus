import math

import numpy as np


def fit_rating(scores, anchors):
    """Elo expected-score fit with fixed opponent ratings; draws count as half."""
    scores, anchors = np.asarray(scores, dtype=float), np.asarray(anchors, dtype=float)
    if scores.shape[-1] != len(anchors) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Each opponent needs a finite score in [0, 1]')
    if not len(anchors) or not np.isfinite(anchors).all():
        raise ValueError('Finite opponent ratings required')
    low = np.full(scores.shape[:-1], anchors.min() - 10000.)
    high = np.full_like(low, anchors.max() + 10000.)
    target = scores.mean(-1)
    for _ in range(60):
        mid = (low + high) / 2
        expected = (1 / (1 + 10 ** ((anchors - mid[..., None]) / 400))).mean(-1)
        low, high = np.where(expected < target, mid, low), np.where(expected >= target, mid, high)
    return np.where(target == 0, -np.inf, np.where(target == 1, np.inf, (low + high) / 2))


def summarize_pool(scores, anchors, samples=10000, seed=0):
    # Shape: opponent, opening pair, colour. Shared openings are one bootstrap cluster.
    scores, anchors = np.asarray(scores, dtype=float), np.asarray(anchors, dtype=float)
    if scores.ndim != 3 or scores.shape[0] != len(anchors) or scores.shape[2] != 2 or not scores.shape[1]:
        raise ValueError('Pool requires equally sized colour-swapped matches')
    if not np.isfinite(scores).all() or not np.isin(scores, [0., .5, 1.]).all() or samples <= 0:
        raise ValueError('Use actual win/draw/loss scores and positive bootstrap samples')
    pairs = scores.mean(-1)
    mean = pairs.mean(-1)
    rating = float(fit_rating(mean, anchors))
    rng = np.random.default_rng(seed)
    indices = rng.integers(pairs.shape[1], size=(samples, pairs.shape[1]))
    boot = fit_rating(pairs[:, indices].mean(-1).T, anchors)
    ordered = np.sort(boot)
    interval = ordered[[int(.025 * (samples - 1)), int(.975 * (samples - 1))]]
    # Applies to this frozen pool's expected score, conditional on its fixed Elo anchors.
    radius = math.sqrt(math.log(40) / (2 * pairs.shape[1]))
    bounds = np.array([max(0., mean.mean() - radius), min(1., mean.mean() + radius)])
    conservative = fit_rating(np.broadcast_to(bounds[:, None], (2, len(anchors))), anchors)
    expected = 1 / (1 + 10 ** ((anchors - rating) / 400))
    finite = lambda value: float(value) if np.isfinite(value) else None
    return {'local_elo': finite(rating), 'elo_opening_bootstrap_95': [finite(v) for v in interval],
            'elo_hoeffding_95': [finite(v) for v in conservative], 'games': int(scores.size),
            'independent_opening_clusters': pairs.shape[1], 'score_by_opponent': mean.tolist(),
            'fitted_score_by_opponent': expected.tolist(), 'score_residuals': (mean - expected).tolist(),
            'rating_pool': 'CCRL-referenced local engine pool; not an official CCRL, FIDE or Lichess rating',
            'uncertainty': 'Intervals are conditional on fixed anchors; hardware, time-control and anchor errors are excluded.'}
