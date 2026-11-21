"""Provenance for validation reports.

A committed report is only useful if it describes the task files that are
actually in the repository. The manifest records the environment a report was
produced in and a content hash of every task, so `python -m forge check-report`
can refuse a report that has gone stale (a task edited, added, or removed
without re-running validation).

Hashes normalize CRLF to LF, so a Windows checkout with automatic line-ending
conversion hashes the same as Linux.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path

import numpy as np

TASK_FILES = ("task.json", "README.md", "grader.py", "reference.py", "starter.py")


def task_files(task_dir: Path) -> list[Path]:
    files = [task_dir / name for name in TASK_FILES if (task_dir / name).exists()]
    return files + sorted((task_dir / "mutants").glob("*.py"))


def task_sha256(task_dir: Path) -> str:
    """One hash over every file that defines a task: names and contents."""
    h = hashlib.sha256()
    for path in task_files(task_dir):
        name = path.relative_to(task_dir).as_posix().encode()
        content = path.read_bytes().replace(b"\r\n", b"\n")
        h.update(name + b"\0" + hashlib.sha256(content).digest())
    return h.hexdigest()


def git_state(root: Path) -> tuple[str | None, bool | None]:
    """(commit, has uncommitted changes), or (None, None) outside a git checkout."""
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None, None
    if head.returncode != 0:
        return None, None
    return head.stdout.strip(), bool(status.stdout.strip())


def build(task_dirs: list[Path], runs: int, root: Path) -> dict:
    commit, dirty = git_state(root)
    return {
        "runs": runs,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "git_commit": commit,
        "git_dirty": dirty,
        "task_sha256": {t.name: task_sha256(t) for t in task_dirs},
    }


def stale_tasks(report: dict, task_dirs: list[Path]) -> list[str]:
    """Tasks whose files differ from what the report recorded, including tasks
    added since the report or recorded in it but since removed."""
    recorded = report.get("manifest", {}).get("task_sha256", {})
    current = {t.name: task_sha256(t) for t in task_dirs}
    return sorted(name for name in current.keys() | recorded.keys()
                  if current.get(name) != recorded.get(name))
