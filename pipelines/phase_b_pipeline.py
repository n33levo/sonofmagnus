#!/usr/bin/env python3
"""
Phase B pipeline: Engine distillation.

Sequence:
    1. Curate "hard" positions from a PGN archive.
    2. Label those positions with Stockfish (or another engine).
    3. Fine-tune the network starting from a previous checkpoint.
    4. Run quick evaluations.

Each major step can run locally or on a remote host over SSH. When a step runs
remotely the script automatically syncs the required inputs to the remote
worktree and retrieves results needed for subsequent local steps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.utils import PipelineRunner, SSHConfig, ensure_paths


# ---------------------------------------------------------------------- #
# step helpers
# ---------------------------------------------------------------------- #

def step_curate(
    runner: PipelineRunner,
    args: argparse.Namespace,
    *,
    remote: bool,
    pgn_path: Path,
    hard_path: Path,
) -> None:
    if remote:
        print("[info] Syncing PGN to remote host for hard-position curation...")
        runner.sync_to_remote(pgn_path)

    if hard_path.exists() and not args.force_curate:
        print(f"[skip] Hard positions already exist: {hard_path}")
        return

    ensure_paths(hard_path)
    cmd = [
        runner.python,
        "scripts/curate_hards.py",
        "--pgn",
        str(pgn_path),
        "--out",
        str(hard_path),
        "--max-games",
        str(args.max_games),
        "--max-positions",
        str(args.max_positions),
        "--min-elo",
        str(args.min_elo),
    ]
    print("[info] Curating hard positions...")
    runner.run(cmd, remote=remote)

    if remote and not args.label_remote:
        print("[info] Fetching hard positions from remote host...")
        runner.fetch_from_remote(hard_path, hard_path)


def step_label(
    runner: PipelineRunner,
    args: argparse.Namespace,
    *,
    remote: bool,
    hard_path: Path,
    labels_path: Path,
) -> None:
    if remote and not args.curate_remote:
        print("[info] Syncing curated hard positions to remote host for labeling...")
        runner.sync_to_remote(hard_path)

    if labels_path.exists() and not args.force_label:
        print(f"[skip] Labels already exist: {labels_path}")
        return

    ensure_paths(labels_path)
    cmd = [
        runner.python,
        "-m",
        "train.distill_labeler",
        "--fens",
        str(hard_path),
        "--out",
        str(labels_path),
        "--engine",
        args.engine_path,
        "--depth",
        str(args.depth),
        "--time",
        str(args.time_limit),
        "--topk",
        str(args.topk),
    ]
    if args.max_label_positions:
        cmd.extend(["--max", str(args.max_label_positions)])
    if args.q_values:
        cmd.append("--q-values")
    print("[info] Generating engine labels...")
    if args.q_values:
        print("[info] Q-value mode enabled (slower but enables multi-task learning)")
    runner.run(cmd, remote=remote)

    if remote and not args.train_remote:
        print("[info] Fetching labeled data from remote host...")
        runner.fetch_from_remote(labels_path, labels_path)


def step_train(
    runner: PipelineRunner,
    args: argparse.Namespace,
    *,
    remote: bool,
    labels_path: Path,
) -> None:
    if remote and not args.label_remote:
        print("[info] Syncing labels to remote host for training...")
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
    if args.q_value_weight > 0:
        cmd.extend(["--q-value-weight", str(args.q_value_weight)])
    print("[info] Fine-tuning on labeled data...")
    if args.q_value_weight > 0:
        print(f"[info] Multi-task training with Q-value weight: {args.q_value_weight}")
    runner.run(cmd, remote=remote)

    if remote:
        latest = Path(args.ckpt_dir or "ckpts") / "latest.pt"
        dest = Path(args.fetch_ckpt or latest)
        print("[info] Fetching updated checkpoint from remote host...")
        runner.fetch_from_remote(latest, dest)


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
        print("[skip] Pytest skipped.")

    if not args.skip_agreement:
        print("[info] Evaluating agreement on engine labels...")
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


# ---------------------------------------------------------------------- #
# CLI plumbing
# ---------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase B (engine distillation) pipeline")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--pgn-path", default="data/lichess_2024-01.pgn.zst")
    parser.add_argument("--hard-path", default="data/hard_positions.jsonl")
    parser.add_argument("--labels-path", default="data/labels_fast.jsonl")
    parser.add_argument("--ckpt-dir", default="ckpts")
    parser.add_argument("--ckpt-path", default="ckpts/latest.pt", help="Checkpoint to evaluate")
    parser.add_argument("--resume-ckpt", default="ckpts/latest.pt", help="Checkpoint to resume training from")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    parser.add_argument("--q-value-weight", type=float, default=0.0, help="Q-value loss weight (beta)")

    parser.add_argument("--max-games", type=int, default=10_000)
    parser.add_argument("--max-positions", type=int, default=50_000)
    parser.add_argument("--min-elo", type=int, default=2000)
    parser.add_argument("--force-curate", action="store_true")

    parser.add_argument("--engine-path", default="/usr/games/stockfish")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--time-limit", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-label-positions", type=int, default=50_000)
    parser.add_argument("--force-label", action="store_true")
    parser.add_argument("--q-values", action="store_true", help="Generate Q-values for multi-task learning")

    parser.add_argument("--model-config", default="main")
    parser.add_argument("--agree-max", type=int, default=2000)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-agreement", action="store_true")

    parser.add_argument("--curate-remote", action="store_true")
    parser.add_argument("--label-remote", action="store_true")
    parser.add_argument("--train-remote", action="store_true")
    parser.add_argument("--ssh-host")
    parser.add_argument("--ssh-user")
    parser.add_argument("--ssh-key")
    parser.add_argument("--remote-workdir")
    parser.add_argument("--fetch-ckpt", help="Local destination for fetched latest.pt (defaults to ckpts/latest.pt)")

    return parser


def build_runner(args: argparse.ArgumentParser.parse_args) -> PipelineRunner:
    use_remote = args.curate_remote or args.label_remote or args.train_remote
    ssh_cfg = None
    if use_remote:
        missing = [
            opt
            for opt, value in {
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

    pgn_path = Path(args.pgn_path)
    hard_path = Path(args.hard_path)
    labels_path = Path(args.labels_path)

    step_curate(runner, args, remote=args.curate_remote, pgn_path=pgn_path, hard_path=hard_path)
    step_label(runner, args, remote=args.label_remote, hard_path=hard_path, labels_path=labels_path)
    step_train(runner, args, remote=args.train_remote, labels_path=labels_path)
    step_evaluate(runner, args, labels_path=labels_path)

    print("[done] Phase B pipeline finished.")


if __name__ == "__main__":
    main()
