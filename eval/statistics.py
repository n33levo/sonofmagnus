import math


def binomial_interval(correct, total, alpha=.05):
    if total <= 0 or not 0 <= correct <= total or not 0 < alpha < 1:
        raise ValueError('Valid counts and confidence level required')
    choose = [math.lgamma(total + 1) - math.lgamma(k + 1) - math.lgamma(total - k + 1) for k in range(total + 1)]

    def inverse(indices, increasing):
        low, high = 0., 1.
        for _ in range(60):
            p = (low + high) / 2
            a, b = math.log(p), math.log1p(-p)
            probability = math.fsum(math.exp(choose[k] + k * a + (total - k) * b) for k in indices)
            if (probability < alpha / 2) == increasing:
                low = p
            else:
                high = p
        return (low + high) / 2

    return [inverse(range(correct, total + 1), True) if correct else 0.,
            inverse(range(correct + 1), False) if correct < total else 1.]
