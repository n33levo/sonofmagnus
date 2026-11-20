"""Check whether the history index ranks completed model matches correctly."""
import argparse
import json
from pathlib import Path

import numpy as np

from model.artifacts import artifact_info


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def compare(rows):
    x = np.array([row['history_quality_index'] for row in rows], float)
    y = np.array([row['external_score'] for row in rows], float)
    if (len(rows) < 3 or len({row['model'] for row in rows}) != len(rows)
            or not np.isfinite([x, y]).all() or np.ptp(x) == 0 or np.ptp(y) == 0
            or ((y < 0) | (y > 1)).any()
            or len({row['external_games'] for row in rows}) != 1):
        raise ValueError('Require distinct, finite models with matched complete-game budgets')
    predicted, baseline = [], []
    for i in range(len(rows)):
        keep = np.arange(len(rows)) != i
        design = np.column_stack((np.ones(keep.sum()), x[keep]))
        weights = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        predicted.append(float(np.clip(np.array([1., x[i]]) @ weights, 0., 1.)))
        baseline.append(float(y[keep].mean()))
    signs = [np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
             for i in range(len(rows)) for j in range(i)]
    return dict(completed_models=len(rows), external_games_per_model=rows[0]['external_games'],
        pearson=float(np.corrcoef(x, y)[0, 1]), spearman=float(np.corrcoef(ranks(x), ranks(y))[0, 1]),
        pair_ordering=dict(agree=signs.count(1), disagree=signs.count(-1), tied=signs.count(0)),
        leave_one_checkpoint_out=dict(index_squared_error=float(np.mean((predicted - y) ** 2)),
            mean_only_squared_error=float(np.mean((baseline - y) ** 2)),
            details=[dict(model=row['model'], actual_external_score=float(score),
                index_estimate=estimate, mean_only_estimate=reference)
                for row, score, estimate, reference in zip(rows, y, predicted, baseline)]),
        offline_elo_conversion_established=False, target_2800_offline_threshold=None,
        automatic_rated=False, promoted=False,
        scope='Descriptive transfer audit on related checkpoints and repeatedly used development openings. '
            'Leaving out one checkpoint does not make its relatives or openings independent. '
            'This tests complete-game score ordering; it cannot establish an official Lichess rating.')


def run(benchmark, output):
    frozen = artifact_info(benchmark)
    result = compare(json.loads(benchmark.read_text())['completed_models'])
    if artifact_info(benchmark) != frozen:
        raise ValueError('Benchmark changed during review')
    result['inputs'] = {str(path.resolve()): artifact_info(path) for path in (benchmark, Path(__file__))}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('inputs', 'leave_one_checkpoint_out')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=Path('out/history-benchmark.json'))
    parser.add_argument('--out', type=Path, default=Path('out/history-transfer.json'))
    args = parser.parse_args()
    run(args.benchmark, args.out)
