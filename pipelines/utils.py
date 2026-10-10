#!/usr/bin/env python3
"""
Shared utilities for chess training pipelines.

Provides helpers to execute commands locally or on a remote host via SSH,
plus convenience wrappers for syncing artefacts between environments.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence


def shlex_join(cmd: Sequence[str]) -> str:
    """Portable shlex.join implementation for Python <3.8."""
    try:
        return shlex.join(cmd)
    except AttributeError:  # pragma: no cover
        return " ".join(shlex.quote(token) for token in cmd)


@dataclass
class SSHConfig:
    host: str
    user: str
    key: str | None = None
    workdir: str | None = None

    def spec(self) -> str:
        return f"{self.user}@{self.host}"


class PipelineRunner:
    """Execute commands for pipeline steps, optionally on a remote host."""

    def __init__(
        self,
        *,
        python: str,
        ssh: SSHConfig | None = None,
    ) -> None:
        self.python = python
        self.ssh = ssh

    # ------------------------------------------------------------------ #
    # core execution helpers
    # ------------------------------------------------------------------ #
    def run(
        self,
        cmd: Sequence[str],
        *,
        remote: bool = False,
        env: Mapping[str, str] | None = None,
        check: bool = True,
    ) -> None:
        if remote:
            if self.ssh is None:
                raise RuntimeError("Remote execution requested but SSH config is missing.")
            cmd = [','.join(str(Path(part).relative_to(Path.cwd())) if Path(part).is_absolute()
                   and Path(part).is_relative_to(Path.cwd()) else part for part in token.split(',')) for token in cmd]
            remote_cmd = shlex_join(cmd)
            if env:
                remote_cmd = "env " + shlex_join([f"{key}={value}" for key, value in env.items()]) + " " + remote_cmd
            pieces: list[str] = ["ssh"]
            if self.ssh.key:
                pieces.extend(["-i", self.ssh.key])
            pieces.append(self.ssh.spec())
            workdir_prefix = ""
            if self.ssh.workdir:
                workdir_prefix = f"cd {shlex.quote(self.ssh.workdir)} && "
            pieces.append(f"{workdir_prefix}{remote_cmd}")
            subprocess.run(pieces, check=check)
        else:
            env_dict: MutableMapping[str, str] = dict(os.environ)
            if env:
                env_dict.update(env)
            subprocess.run(cmd, check=check, env=env_dict)

    # ------------------------------------------------------------------ #
    # rsync helpers
    # ------------------------------------------------------------------ #
    def _ensure_ssh(self) -> SSHConfig:
        if self.ssh is None:
            raise RuntimeError("SSH configuration required for remote sync.")
        if not self.ssh.workdir:
            raise RuntimeError("Remote workdir must be configured for sync operations.")
        return self.ssh

    def _rsync_base(self) -> list[str]:
        cmd: list[str] = ["rsync", "-azP"]
        if self.ssh and self.ssh.key:
            cmd.extend(["-e", f"ssh -i {shlex.quote(self.ssh.key)}"])
        return cmd

    def sync_to_remote(self, local_path: Path) -> None:
        ssh = self._ensure_ssh()
        if not local_path.exists():
            raise FileNotFoundError(local_path)

        local_relative = local_path.resolve().relative_to(Path.cwd())
        remote_dir = Path(ssh.workdir) / local_relative.parent
        mkdir_cmd: list[str] = ["ssh"]
        if ssh.key:
            mkdir_cmd.extend(["-i", ssh.key])
        mkdir_cmd.append(ssh.spec())
        mkdir_cmd.append(f"mkdir -p {shlex.quote(str(remote_dir))}")
        subprocess.run(mkdir_cmd, check=True)

        rsync_cmd = self._rsync_base()
        rsync_cmd.append(str(local_path))
        rsync_cmd.append(f"{ssh.spec()}:{shlex.quote(str(remote_dir))}/")
        subprocess.run(rsync_cmd, check=True)

    def fetch_from_remote(self, remote_relative: Path, destination: Path) -> None:
        ssh = self._ensure_ssh()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if remote_relative.is_absolute():
            remote_relative = remote_relative.relative_to(Path.cwd())
        remote_path = Path(ssh.workdir) / remote_relative
        rsync_cmd = self._rsync_base()
        rsync_cmd.append(f"{ssh.spec()}:{shlex.quote(str(remote_path))}")
        rsync_cmd.append(str(destination))
        subprocess.run(rsync_cmd, check=True)

    def sync_repository(self):
        ssh = self._ensure_ssh()
        self.sync_to_remote(Path("requirements.txt"))
        cmd = self._rsync_base()
        for path in (".git", ".venv", "data", "ckpts", "out", "__pycache__", ".pytest_cache"):
            cmd += ["--exclude", path]
        subprocess.run([*cmd, "./", f"{ssh.spec()}:{shlex.quote(ssh.workdir)}/"], check=True)
        self.run(["python3", "-m", "venv", ".venv"], remote=True)
        self.run([self.python, "-m", "pip", "install", "-r", "requirements.txt"], remote=True)


def ensure_paths(*paths: Path) -> None:
    for path in paths:
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
