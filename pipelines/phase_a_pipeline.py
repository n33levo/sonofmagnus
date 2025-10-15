#!/usr/bin/env python3
"""
Phase A pipeline: Behavioral cloning from human games.

Run a single script to:
    1. Download a Lichess PGN archive (optional if already cached).
    2. Convert that PGN into supervised JSONL training data.
    3. Train the neural network (locally or via SSH on a GPU box).
    4. Execute sanity tests and quick evaluations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from pipelines.utils import PipelineRunner, SSHConfig, ensure_paths


# ---------------------------------------------------------------------- #
# pipeline steps
# ---------------------------------------------------------------------- #

def step_download_pgn(runner: PipelineRunner, args: argparse.Namespace, pgn_path: Path) -> None:
    if args.skip_download and pgn_path.exists():
        print(f"[skip] PGN download skipped ({pgn_path})")
        return

    if pgn_path.exists() and not args.force_download:
        print(f"[skip] PGN already present at {pgn_path}")
        return

    ensure_paths(pgn_path.parent)
    env = {}
    if args.lichess_month:
        env["YEAR_MONTH"] = args.lichess_month
    print("[info] Downloading Lichess archive via scripts/download_lichess.sh ...")
    runner.run(["bash", "scripts/download_lichess.sh"], env=env)


def step_build_dataset(runner: PipelineRunner, args: argparse.Namespace, pgn_path: Path, dataset_path: Path) -> None:
    if dataset_path.exists() and not args.force_rebuild_dataset:
        print(f"[skip] Dataset already built: {dataset_path}")
        return

    ensure_paths(dataset_path)
    cmd = [
        runner.python,
        "-m",
        "train.dataset",
        "--pgn",
        str(pgn_path),
        "--out",
        str(dataset_path),
        "--elo-min",
        str(args.elo_min),
        "--sample-rate",
        str(args.sample_rate),
        "--max-positions",
        str(args.max_positions),
    ]
    print("[info] Building supervised dataset...")
    runner.run(cmd)


def step_train(runner: PipelineRunner, args: argparse.Namespace, dataset_path: Path) -> None:
    cmd = [
        runner.python,
        "-m",
        "train.train",
        "--config",
        args.config,
        "--data",
        str(dataset_path),
        "--epochs",
        str(args.epochs),
        "--amp",
        args.amp,
    ]
    if args.max_samples:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.ckpt_dir:
        cmd.extend(["--ckpt-dir", args.ckpt_dir])

    if args.train_remote:
        if args.sync_dataset:
            print("[info] Syncing dataset to remote host...")
            runner.sync_to_remote(dataset_path)
        print("[info] Running training on remote host...")
        runner.run(cmd, remote=True)
        if args.fetch_ckpt:
            remote_latest = Path(args.ckpt_dir) / "latest.pt"
            print("[info] Fetching remote checkpoint...")
            runner.fetch_from_remote(remote_latest, Path(args.fetch_ckpt))
    else:
        print("[info] Running training locally...")
        runner.run(cmd)


def step_tests_and_eval(
    runner: PipelineRunner,
    args: argparse.Namespace,
    agreement_paths: Iterable[Path],
) -> None:
    if not args.skip_tests:
        print("[info] Running pytest suite...")
        runner.run([runner.python, "-m", "pytest", "tests", "-q"])
    else:
        print("[skip] Pytest suite skipped.")

    if not args.skip_agreement:
        for label_path in agreement_paths:
            if not label_path.exists():
                print(f"[warn] Skipping agreement evaluation; labels missing: {label_path}")
                continue
            print(f"[info] Evaluating agreement on {label_path} ...")
            runner.run(
                [
                    runner.python,
                    "-m",
                    "eval.agree",
                    "--ckpt",
                    args.ckpt_path,
                    "--labels",
                    str(label_path),
                    "--config",
                    args.model_config,
                    "--max",
                    str(args.agree_max),
                ]
            )
    else:
        print("[skip] Agreement evaluation skipped.")

    puzzles_file = Path(args.puzzles_path)
    if args.skip_puzzles:
        print("[skip] Puzzle evaluation skipped.")
    elif puzzles_file.exists():
        print(f"[info] Evaluating puzzles on {puzzles_file} ...")
        runner.run(
            [
                runner.python,
                "-m",
                "eval.puzzles",
                "--ckpt",
                args.ckpt_path,
                "--puzzles",
                str(puzzles_file),
                "--config",
                args.model_config,
                "--max",
                str(args.puzzles_max),
            ]
        )
    else:
        print(f"[warn] Puzzles file not found ({puzzles_file}); skipping.")


# ---------------------------------------------------------------------- #
# argument parsing
# ---------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase A (behavioral cloning) pipeline")
    parser.add_argument("--python", default=".venv/bin/python", help="Python executable to use")
    parser.add_argument("--config", default="configs/main.yaml", help="Training config path")
    parser.add_argument("--pgn-path", default="data/lichess_2024-01.pgn.zst")
    parser.add_argument("--dataset-path", default="data/positions.jsonl")
    parser.add_argument("--ckpt-dir", default="ckpts")
    parser.add_argument("--ckpt-path", default="ckpts/latest.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    parser.add_argument("--elo-min", type=int, default=1800)
    parser.add_argument("--sample-rate", type=float, default=0.1)
    parser.add_argument("--max-positions", type=int, default=1_000_000)
    parser.add_argument("--max-samples", type=int, help="Optional cap for training samples")
    parser.add_argument("--model-config", default="main", help="Config key for eval scripts")
    parser.add_argument("--puzzles-path", default="data/puzzles.jsonl")
    parser.add_argument("--puzzles-max", type=int, default=1000)
    parser.add_argument("--agree-max", type=int, default=2000)

    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--lichess-month", help="Override YEAR_MONTH passed to download script (YYYY-MM)")
    parser.add_argument("--force-rebuild-dataset", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-agreement", action="store_true")
    parser.add_argument("--skip-puzzles", action="store_true")

    # Remote execution options
    parser.add_argument("--train-remote", action="store_true", help="Run training on remote host via SSH")
    parser.add_argument("--ssh-host")
    parser.add_argument("--ssh-user")
    parser.add_argument("--ssh-key")
    parser.add_argument("--remote-workdir", help="Remote repository path")
    parser.add_argument("--sync-dataset", action="store_true", help="Sync dataset file to remote before training")
    parser.add_argument("--fetch-ckpt", help="Local path to store fetched latest.pt after remote training")

    return parser


def build_runner(args: argparse.Namespace) -> PipelineRunner:
    ssh_cfg = None
    if args.train_remote:
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
    dataset_path = Path(args.dataset_path)

    step_download_pgn(runner, args, pgn_path)
    step_build_dataset(runner, args, pgn_path, dataset_path)
    step_train(runner, args, dataset_path)
    step_tests_and_eval(runner, args, [dataset_path])

    print("[done] Phase A pipeline finished.")


if __name__ == "__main__":
    main()
