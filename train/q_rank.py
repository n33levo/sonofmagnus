import mlx.core as mx


def ranking_loss(policy, target, known, weight=None):
    policy = mx.where(known, policy, 0.)
    best = mx.max(mx.where(known, target, -1.), axis=-1, keepdims=True)
    winners = known & (target == best)
    anchor = (policy * winners).sum(-1, keepdims=True) / mx.maximum(winners.sum(-1, keepdims=True), 1)
    regret = mx.where(known, best - target, 0.)
    total = regret.sum(-1)
    terms = mx.logaddexp(mx.array(0., dtype=mx.float32), policy - anchor)
    rows = (terms * regret).sum(-1) / mx.maximum(total, 1e-8)
    quality = mx.ones_like(total) if weight is None else weight
    labelled = quality * (total > 0)
    return (rows * labelled).sum() / mx.maximum(labelled.sum(), 1e-8)
