#!/usr/bin/env python3
"""
Phase C pipeline: Self-play relabeling (DAgger-lite).

Steps:
    1. Generate self-play games with the current checkpoint.
    2. Relabel the visited positions using Stockfish.
    3. Fine-tune on the self-play labels.
    4. Run smoke-test evaluations.

Each stage can run locally or via SSH on a GPU host.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.utils import PipelineRunner, SSHConfig


def step_selfplay(
    runner: PipelineRunner,
    args: argparse.Namespace,
    *,
    remote: bool,
    checkpoint_path: Path,
    selfplay_path: Path,
) -> None:
    if remote:
        print("[info] Syncing checkpoint to remote host for self-play...")
        runner.sync_to_remote(checkpoint_path)

    cmd = [
        runner.python,
        "-m",
        "train.selfplay",
        "--ckpt",
        str(checkpoint_path),
        "--out",
        str(selfplay_path),
        "--games",
        str(args.games),
        "--config",
        args.model_config,
        "--max-moves",
        str(args.max_moves),
        "--device",
        args.selfplay_device,
    ]
    print("[info] Generating self-play games...")
    runner.run(cmd, remote=remote)

    if remote and not args.label_remote:
        print("[info] Fetching self-play trajectories from remote host...")
        runner.fetch_from_remote(selfplay_path, selfplay_path)


def step_label(
    runner: PipelineRunner,
    args: argparse.Namespace,
    *,
    remote: bool,
    selfplay_path: Path,
    labels_path: Path,
) -> None:
    if remote and not args.selfplay_remote:
        print("[info] Syncing self-play data to remote host for labeling...")
        runner.sync_to_remote(selfplay_path)

    cmd = [
        runner.python,
        "-m",
        "train.distill_labeler",
        "--fens",
        str(selfplay_path),
        "--out",
        str(labels_path),
        "--engine",
        args.engine_path,
        "--preset",
        args.label_preset,
    ]
    if args.max_label_positions:
        cmd.extend(["--max", str(args.max_label_positions)])
    print("[info] Labeling self-play positions with engine guidance...")
    runner.run(cmd, remote=remote)

    if remote and not args.train_remote:
        print("[info] Fetching labeled self-play data from remote host...")
        runner.fetch_from_remote(labels_path, labels_path)


def step_train(
    runner: PipelineRunner,
    args: argparse.Namespace,
    *,
    remote: bool,
    labels_path: Path,
) -> None:
    if remote and not args.label_remote:
        print("[info] Syncing labels to remote host for fine-tuning...")
        runner.sync_to_remote(labels_path)

    cmd = [
        runner.python,
        "-m",
        "train.train",
        "--config",
        args.config,
        "--data",
        str(labels_path),
        "--epochs",
        str(args.epochs),
        "--amp",
        args.amp,
        "--resume",
        args.resume_ckpt,
    ]
    if args.ckpt_dir:
        cmd.extend(["--ckpt-dir", args.ckpt_dir])

    print("[info] Fine-tuning on self-play labels...")
    runner.run(cmd, remote=remote)

    if remote:
        latest = Path(args.ckpt_dir or "ckpts") / "latest.pt"
        destination = Path(args.fetch_ckpt or latest)
        print("[info] Fetching updated checkpoint from remote host...")
        runner.fetch_from_remote(latest, destination)


def step_evaluate(
    runner: PipelineRunner,
    args: argparse.Namespace,
    *,
    labels_path: Path,
) -> None:
    if not args.skip_tests:
        print("[info] Running pytest...")
        runner.run([runner.python, "-m", "pytest", "tests", "-q"])
    else:
        print("[skip] Test suite skipped.")

    if not args.skip_agreement:
        print("[info] Evaluating agreement on self-play labels...")
        runner.run(
            [
                runner.python,
                "-m",
                "eval.agree",
                "--ckpt",
                args.ckpt_path,
                "--labels",
                str(labels_path),
                "--config",
                args.model_config,
                "--max",
                str(args.agree_max),
            ]
        )
    else:
        print("[skip] Agreement evaluation skipped.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase C (self-play relabel) pipeline")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--model-config", default="main")
    parser.add_argument("--ckpt-dir", default="ckpts")
    parser.add_argument("--ckpt-path", default="ckpts/latest.pt")
    parser.add_argument("--resume-ckpt", default="ckpts/latest.pt")
    parser.add_argument("--selfplay-path", default="data/selfplay.jsonl")
    parser.add_argument("--labels-path", default="data/selfplay_labels.jsonl")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--selfplay-device", default="cuda", help="Device argument passed to train.selfplay")
    parser.add_argument("--engine-path", default="/usr/games/stockfish")
    parser.add_argument("--label-preset", default="fast", help="Preset passed to distill_labeler")
    parser.add_argument("--max-label-positions", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    parser.add_argument("--agree-max", type=int, default=2000)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-agreement", action="store_true")

    parser.add_argument("--selfplay-remote", action="store_true")
    parser.add_argument("--label-remote", action="store_true")
    parser.add_argument("--train-remote", action="store_true")
    parser.add_argument("--ssh-host")
    parser.add_argument("--ssh-user")
    parser.add_argument("--ssh-key")
    parser.add_argument("--remote-workdir")
    parser.add_argument("--fetch-ckpt", help="Local destination for fetched latest.pt after training")

    return parser


def build_runner(args: argparse.Namespace) -> PipelineRunner:
    use_remote = args.selfplay_remote or args.label_remote or args.train_remote
    ssh_cfg = None
    if use_remote:
        missing = [
            option
            for option, value in {
                "--ssh-host": args.ssh_host,
                "--ssh-user": args.ssh_user,
                "--remote-workdir": args.remote_workdir,
            }.items()
            if not value
        ]
        if missing:
            raise SystemExit(f"Missing SSH configuration: {', '.join(missing)}")
        ssh_cfg = SSHConfig(
            host=args.ssh_host,
            user=args.ssh_user,
            key=args.ssh_key,
            workdir=args.remote_workdir,
        )
    return PipelineRunner(python=args.python, ssh=ssh_cfg)


def main() -> None:
    args = build_parser().parse_args()
    runner = build_runner(args)

    checkpoint_path = Path(args.ckpt_path)
    selfplay_path = Path(args.selfplay_path)
    labels_path = Path(args.labels_path)

    step_selfplay(runner, args, remote=args.selfplay_remote, checkpoint_path=checkpoint_path, selfplay_path=selfplay_path)
    step_label(runner, args, remote=args.label_remote, selfplay_path=selfplay_path, labels_path=labels_path)
    step_train(runner, args, remote=args.train_remote, labels_path=labels_path)
    step_evaluate(runner, args, labels_path=labels_path)
    print("[done] Phase C pipeline finished.")


if __name__ == "__main__":
    main()
