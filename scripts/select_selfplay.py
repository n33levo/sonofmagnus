import argparse
import heapq
import json
import math
from pathlib import Path
import random

from chessio.records import holdout_positions
from model.artifacts import artifact_info


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
    blocked, seen, heap = holdout_positions(exclude), set(), []
    rng, count = random.Random(seed), 0
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            entropy = float(row['policy_entropy'])
            if not math.isfinite(entropy) or not 0 <= entropy <= 1:
                raise ValueError('Invalid policy entropy')
            count += 1
            fields = row['fen'].split()
            key = ' '.join(fields[:5])
            if key in seen or ' '.join(fields[:4]) in blocked:
                continue
            seen.add(key)
            candidate = entropy, rng.random(), line
            if len(heap) < budget:
                heapq.heappush(heap, candidate)
            elif candidate[:2] > heap[0][:2]:
                heapq.heapreplace(heap, candidate)
    if count != metadata['positions'] or len(heap) != budget or artifact_info(source) != frozen:
        raise ValueError('Wrong source count, insufficient distinct positions, or changed source')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.writelines(row[2] for row in sorted(heap, reverse=True))
    result = dict(source=frozen, artifact=artifact_info(output), input_positions=count,
                  eligible_distinct_positions=len(seen), selected_positions=budget, fraction=budget / count,
                  seed=seed, criterion='Highest normalized legal-policy entropy; seeded tie-breaking',
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
