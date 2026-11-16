"""Runs inside a fresh, isolated interpreter (python -I): loads one solution
and one task's hidden grader, runs every check, prints a single marked JSON
line. Kept separate from the parent so a crashing, hanging, or
state-mutating solution can never affect the harness or other runs."""

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forge.checks import collect  # noqa: E402

MARKER = "FORGE_RESULT:"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _emit(payload: dict) -> None:
    sys.stdout.write("\n" + MARKER + json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> None:
    task_dir, solution_path = Path(sys.argv[1]), Path(sys.argv[2])
    try:
        solution = _load("solution", solution_path)
    except BaseException as e:  # syntax errors, import errors, SystemExit
        _emit({"status": "import_error", "error": f"{type(e).__name__}: {e}", "checks": []})
        return
    grader = _load("forge_grader", task_dir / "grader.py")

    results = []
    for meta, fn in collect(grader):
        t0 = time.perf_counter()
        try:
            fn(solution)
            passed, message = True, ""
        except AssertionError as e:
            passed, message = False, str(e) or "assertion failed"
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            passed, message = False, f"{type(e).__name__}: {e} (line {tb.lineno} of {Path(tb.filename).name})"
        results.append({**meta, "passed": passed, "message": message[:500],
                        "seconds": round(time.perf_counter() - t0, 4)})
    _emit({"status": "ok", "checks": results})


if __name__ == "__main__":
    main()
