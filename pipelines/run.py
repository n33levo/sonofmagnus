import argparse
from pathlib import Path

from pipelines.utils import PipelineRunner, SSHConfig


def main(phase):
    parser = argparse.ArgumentParser(description=f'Phase {phase}')
    parser.add_argument('--python', default='.venv/bin/python')
    parser.add_argument('--config', default='configs/main.yaml')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--pgn-path', default='data/lichess_2024-01.pgn.zst')
    parser.add_argument('--dataset-path', default='data/positions.jsonl')
    parser.add_argument('--hard-path', default='data/hard_positions.jsonl')
    parser.add_argument('--labels-path', default=f'data/phase_{phase.lower()}_labels.jsonl')
    parser.add_argument('--selfplay-path', default='data/selfplay.jsonl')
    parser.add_argument('--ckpt-dir', default='ckpts')
    parser.add_argument('--ckpt-path')
    parser.add_argument('--resume-ckpt')
    parser.add_argument('--val-data')
    parser.add_argument('--replay-data')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--amp', choices=['bf16', 'fp16', 'none'], default='none')
    parser.add_argument('--q-value-weight', type=float, default=.4)
    parser.add_argument('--q-values', action='store_true')
    parser.add_argument('--elo-min', '--min-elo', dest='min_elo', type=int, default=2000)
    parser.add_argument('--sample-rate', type=float, default=.1)
    parser.add_argument('--max-positions', type=int, default=50_000)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--max-games', type=int, default=10_000)
    parser.add_argument('--engine-path', default='stockfish')
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--time-limit', type=float, default=.05)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--max-label-positions', type=int, default=50_000)
    parser.add_argument('--label-preset', choices=['fast', 'deep'], default='fast')
    parser.add_argument('--games', type=int, default=2000)
    parser.add_argument('--max-moves', type=int, default=400)
    parser.add_argument('--selfplay-device', default='auto')
    parser.add_argument('--model-config', default='main')
    parser.add_argument('--puzzles-path', default='data/puzzles.jsonl')
    parser.add_argument('--puzzles-max', type=int, default=1000)
    parser.add_argument('--agree-max', type=int, default=2000)
    parser.add_argument('--lichess-month', default='2024-01')
    parser.add_argument('--ssh-host')
    parser.add_argument('--ssh-user')
    parser.add_argument('--ssh-key')
    parser.add_argument('--remote-workdir')
    parser.add_argument('--fetch-ckpt')
    for flag in ('skip-download', 'force-download', 'force-rebuild-dataset', 'force-curate',
                 'force-label', 'skip-tests', 'skip-agreement', 'skip-puzzles', 'train-remote',
                 'curate-remote', 'label-remote', 'selfplay-remote', 'sync-dataset'):
        parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    args.force_label = args.force_label or args.force_curate
    remote = any((args.train_remote, args.curate_remote, args.label_remote, args.selfplay_remote))
    if remote and not all((args.ssh_host, args.ssh_user, args.remote_workdir)):
        parser.error('Remote stages require --ssh-host, --ssh-user and --remote-workdir')
    runner = PipelineRunner(python=args.python, ssh=SSHConfig(args.ssh_host, args.ssh_user,
                           args.ssh_key, args.remote_workdir) if remote else None)
    if remote:
        runner.sync_repository()
    latest = Path(args.ckpt_dir) / 'latest.pt'
    checkpoint = Path(args.ckpt_path or args.fetch_ckpt or latest)
    resume = Path(args.resume_ckpt or checkpoint)

    def execute(module, flags, remote=False, inputs=(), outputs=()):
        if remote:
            for path in inputs:
                runner.sync_to_remote(Path(path))
        runner.run([args.python, '-m', module, *map(str, flags)], remote=remote)
        if remote:
            for path in outputs:
                runner.fetch_from_remote(Path(path), Path(path))

    if phase == 'A':
        data = Path(args.dataset_path)
        if not data.exists() or args.force_rebuild_dataset:
            pgn = Path(args.pgn_path)
            if not pgn.exists() or args.force_download:
                if args.skip_download:
                    parser.error(f'PGN does not exist: {pgn}')
                runner.run(['bash', 'scripts/download_lichess.sh'], env={
                    'YEAR_MONTH': args.lichess_month, 'OUTPUT_FILE': str(pgn),
                    'FORCE_DOWNLOAD': '1' if args.force_download else '0'})
            execute('train.dataset', ['--pgn', pgn, '--out', data, '--elo-min', args.min_elo,
                    '--sample-rate', args.sample_rate, '--max-positions', args.max_positions])
    else:
        data = Path(args.labels_path)
        if phase == 'C':
            source = Path(args.selfplay_path)
            execute('train.selfplay', ['--ckpt', resume, '--out', source, '--games', args.games,
                    '--config', args.model_config, '--max-moves', args.max_moves, '--device', args.selfplay_device],
                    args.selfplay_remote, [resume], [source])
        else:
            source = Path(args.hard_path)
            if (not source.exists() or args.force_curate) and (not data.exists() or args.force_label):
                execute('scripts.curate_hards', ['--pgn', args.pgn_path, '--out', source,
                        '--max-games', args.max_games, '--max-positions', args.max_positions, '--min-elo', args.min_elo],
                        args.curate_remote, [args.pgn_path], [source])
        if phase == 'C' or not data.exists() or args.force_label:
            flags = ['--fens', source, '--out', data, '--engine', args.engine_path,
                     '--depth', args.depth, '--time', args.time_limit, '--topk', args.topk,
                     '--max', args.max_label_positions]
            if phase == 'C':
                flags += ['--preset', args.label_preset]
            if args.q_values:
                flags += ['--q-values']
            execute('train.distill_labeler', flags, args.label_remote, [source], [data])
    flags = ['--config', args.config, '--data', str(data) + (',' + args.replay_data if args.replay_data else ''),
             '--epochs', args.epochs, '--amp', args.amp, '--device', args.device, '--ckpt-dir', args.ckpt_dir]
    inputs = [data]
    if args.replay_data:
        inputs += [args.replay_data]
    if phase != 'A' or args.resume_ckpt:
        flags += ['--resume', resume]
        inputs += [resume]
    if args.q_values:
        flags += ['--q-value-weight', args.q_value_weight]
    if args.batch_size:
        flags += ['--batch-size', args.batch_size]
    if args.max_samples:
        flags += ['--max-samples', args.max_samples]
    if args.val_data:
        flags += ['--val-data', args.val_data]
        inputs += [args.val_data]
    execute('train.train', flags, args.train_remote, inputs, [latest])
    if args.fetch_ckpt and latest != checkpoint:
        import shutil
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(latest, checkpoint)
    else:
        checkpoint = latest
    if not args.skip_tests:
        runner.run([args.python, '-m', 'pytest', 'tests', '-q'])
    if args.val_data and not args.skip_agreement:
        execute('eval.agree', ['--ckpt', checkpoint, '--labels', args.val_data, '--device', args.device,
                              '--max', args.agree_max])
    if not args.skip_puzzles and Path(args.puzzles_path).exists():
        execute('eval.puzzles', ['--ckpt', checkpoint, '--puzzles', args.puzzles_path,
                                '--device', args.device, '--max', args.puzzles_max])
    print(f'Phase {phase}: {checkpoint}')
