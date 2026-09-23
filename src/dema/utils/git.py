from __future__ import annotations

import subprocess

from .config import REPO_ROOT


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_commit() -> str | None:
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool | None:
    status = _git("status", "--porcelain", "--untracked-files=no")
    return None if status is None else bool(status)
