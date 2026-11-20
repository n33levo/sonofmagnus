import argparse
import json
from pathlib import Path
import subprocess
import sys

from model.artifacts import artifact_info


def main():
    parser = argparse.ArgumentParser(description='On-policy self-play, selective engine labels and cumulative replay')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--config', default='configs/main.yaml')
    parser.add_argument('--replay', required=True)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--exclude-data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--positions', type=int, default=200000, help='Positions per iteration')
    parser.add_argument('--fraction', type=float, default=.15)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--amp', choices=['none', 'fp16', 'bf16'], default='fp16')
    args = parser.parse_args()
    if min(args.iterations, args.positions, args.batch_size, args.workers, args.depth) < 1 or not 0 < args.fraction <= 1 or args.lr <= 0:
        parser.error('Positive budgets and learning rate, fraction in (0,1] required')
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    checkpoint, labels = args.ckpt, []
    state = dict(arguments=vars(args), initial_checkpoint=artifact_info(checkpoint),
                 replay=artifact_info(args.replay), engine=artifact_info(args.engine),
                 producer=artifact_info(__file__), completed_iterations=[])
    (root / 'run.json').write_text(json.dumps(state, indent=2) + '\n')

    def run(module, *arguments):
        command = [sys.executable, '-m', module, *map(str, arguments)]
        print(json.dumps({'command': command}), flush=True)
        subprocess.run(command, check=True)

    for iteration in range(1, args.iterations + 1):
        directory = root / f'iteration-{iteration}'
        directory.mkdir()
        corpus, selected, labeled = [directory / name for name in ('selfplay.jsonl', 'selected.jsonl', 'labels.jsonl')]
        run('train.selfplay', '--ckpt', checkpoint, '--out', corpus, '--positions', args.positions,
            '--games', max(20000, args.positions), '--batch-size', 32, '--seed', args.seed + iteration,
            '--device', args.device)
        run('scripts.check_selfplay', '--source', corpus, '--ckpt', checkpoint, '--out', directory / 'selfplay-review.json')
        run('scripts.select_selfplay', '--source', corpus, '--out', selected, '--fraction', args.fraction,
            '--seed', args.seed + iteration, '--exclude-data', args.exclude_data)
        run('scripts.relabel_selfplay', '--source', selected, '--out', labeled,
            '--engine', args.engine, '--depth', args.depth, '--workers', args.workers)
        run('scripts.check_relabels', '--source', selected, '--labels', labeled, '--out', directory / 'label-review.json')
        labels.append(str(labeled))
        run('train.train', '--config', args.config, '--resume', checkpoint,
            '--data', ','.join([args.replay, *labels]), '--data-weights', ','.join(['1', *['3'] * len(labels)]),
            '--exclude-data', args.exclude_data, '--epochs', 1, '--lr', args.lr, '--q-value-weight', .4,
            '--batch-size', args.batch_size, '--workers', args.workers, '--device', args.device,
            '--amp', args.amp, '--seed', args.seed + iteration, '--ckpt-dir', directory / 'ckpts')
        checkpoint = str(directory / 'ckpts/latest.pt')
        state['completed_iterations'].append(dict(iteration=iteration, checkpoint=artifact_info(checkpoint),
                                                  labeled=artifact_info(labeled)))
        (root / 'run.json').write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps({'final_checkpoint': checkpoint, 'iterations': args.iterations}), flush=True)


if __name__ == '__main__':
    main()
