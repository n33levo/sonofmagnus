import argparse
from collections import Counter
import heapq
import json
import math
from pathlib import Path
import random

from chessio.records import holdout_positions
from model.artifacts import artifact_info


def phase(row):
    pieces = sum(letter.isalpha() for letter in row['fen'].split()[0])
    return 0 if pieces >= 24 else 1 if pieces >= 16 else 2 if pieces >= 8 else 3


def select(source, output, fraction=.15, seed=0, exclude=()):
    source, output = Path(source), Path(output)
    if not 0 < fraction <= 1 or output.exists():
        raise ValueError('Use a positive fraction at most one and a new output path')
    frozen = artifact_info(source)
    metadata = json.loads(source.with_suffix('.json').read_text())
    if metadata['artifact'] != frozen:
        raise ValueError('Self-play source differs from its manifest')
    if not isinstance(metadata['positions'], int) or metadata['positions'] <= 0:
        raise ValueError('Self-play position count must be a positive integer')
    budget = math.ceil(metadata['positions'] * fraction)
    blocked = holdout_positions(exclude)

    def eligible():
        seen, count = set(), 0
        with source.open() as stream:
            for count, line in enumerate(stream, 1):
                row = json.loads(line)
                entropy = float(row['policy_entropy'])
                if not math.isfinite(entropy) or not 0 <= entropy <= 1:
                    raise ValueError('Invalid policy entropy')
                fields = row['fen'].split()
                key = ' '.join(fields[:5])
                if key in seen or ' '.join(fields[:4]) in blocked:
                    continue
                seen.add(key)
                yield row, line
        if count != metadata['positions']:
            raise ValueError('Wrong source count')

    counts = Counter(phase(row) for row, _ in eligible())
    total = sum(counts.values())
    if total < budget:
        raise ValueError('Insufficient distinct positions')
    quotas = {group: budget * size // total for group, size in counts.items()}
    remainder = sorted(counts, key=lambda group: (budget * counts[group] % total, -group), reverse=True)
    for group in remainder[:budget - sum(quotas.values())]:
        quotas[group] += 1
    rng, heaps = random.Random(seed), {group: [] for group in counts}
    for row, line in eligible():
        group = phase(row)
        heap, quota = heaps[group], quotas[group]
        candidate = float(row['policy_entropy']), rng.random(), line
        if len(heap) < quota:
            heapq.heappush(heap, candidate)
        elif quota and candidate[:2] > heap[0][:2]:
            heapq.heapreplace(heap, candidate)
    selected = [row for heap in heaps.values() for row in heap]
    if len(selected) != budget or artifact_info(source) != frozen:
        raise ValueError('Wrong source count, insufficient distinct positions, or changed source')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.writelines(row[2] for row in sorted(selected, reverse=True))
    result = dict(source=frozen, artifact=artifact_info(output), input_positions=metadata['positions'],
                  eligible_distinct_positions=total, selected_positions=budget, fraction=budget / metadata['positions'],
                  seed=seed, criterion='Phase-proportional highest normalized legal-policy entropy; seeded tie-breaking',
                  phase_piece_thresholds=[24, 16, 8], eligible_by_phase=dict(counts), selected_by_phase=quotas,
                  excluded_inputs=list(map(str, exclude)), producer_sha256=artifact_info(__file__)['sha256'])
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--fraction', type=float, default=.15)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exclude-data', default='')
    args = parser.parse_args()
    print(json.dumps(select(args.source, args.out, args.fraction, args.seed, args.exclude_data.split(',') if args.exclude_data else ()), indent=2))
