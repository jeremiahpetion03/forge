"""Grade one solution against one task in an isolated subprocess.

Isolation here means a separate interpreter with a timeout, a scratch working
directory, and no user site-packages or PYTHONPATH. It protects the harness
from crashes, hangs, and leaked state. It is NOT a security sandbox: do not
run untrusted code outside a container or VM.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

CHILD = Path(__file__).resolve().parent / "_child.py"
MARKER = "FORGE_RESULT:"
ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"


@dataclass
class RunResult:
    status: str  # ok | import_error | timeout | crashed
    checks: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def all_passed(self) -> bool:
        return self.status == "ok" and bool(self.checks) and all(c["passed"] for c in self.checks)

    @property
    def score(self) -> float:
        total = sum(c["weight"] for c in self.checks)
        if self.status != "ok" or not total:
            return 0.0
        return sum(c["weight"] for c in self.checks if c["passed"]) / total

    @property
    def failed(self) -> list[str]:
        return [c["name"] for c in self.checks if not c["passed"]]

    def pass_vector(self) -> tuple:
        return (self.status, tuple(c["passed"] for c in self.checks))


def task_path(task: str | Path) -> Path:
    p = Path(task)
    return p if p.is_dir() else TASKS_DIR / str(task)


def time_limit(task_dir: Path) -> float:
    return float(json.loads((task_dir / "task.json").read_text(encoding="utf-8")).get("time_limit_s", 60))


def run(task: str | Path, solution: str | Path, timeout: float | None = None) -> RunResult:
    task_dir = task_path(task)
    solution = Path(solution).resolve()
    timeout = timeout if timeout is not None else time_limit(task_dir)
    cmd = [sys.executable, "-I", str(CHILD), str(task_dir.resolve()), str(solution)]
    with tempfile.TemporaryDirectory() as scratch:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=scratch)
        except subprocess.TimeoutExpired:
            return RunResult("timeout", error=f"exceeded {timeout:g}s time limit")

    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(MARKER)]
    if not lines:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        return RunResult("crashed", error=f"exit code {proc.returncode}: " + " | ".join(tail))
    payload = json.loads(lines[-1][len(MARKER):])
    return RunResult(payload["status"], payload.get("checks", []), payload.get("error", ""))
