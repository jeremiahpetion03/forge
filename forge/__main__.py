"""Forge command line.

    python -m forge list                          list tasks
    python -m forge grade <task> <solution.py>    grade a solution
    python -m forge validate [--runs 3]           validate every task, write VALIDATION.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runner import ROOT, run, task_path
from .validate import list_tasks, validate_all, write_reports


def cmd_list(_args) -> int:
    for t in list_tasks():
        meta = json.loads((t / "task.json").read_text(encoding="utf-8"))
        print(f"{t.name:28s} [{meta['difficulty']}] {meta['title']}")
    return 0


def cmd_grade(args) -> int:
    result = run(task_path(args.task), args.solution)
    if result.status != "ok":
        print(f"{result.status.upper()}: {result.error}")
        return 1
    for c in result.checks:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  {mark}  [{c['kind']}] {c['name']}" + (f"\n        {c['message']}" if c["message"] else ""))
    print(f"\nscore {result.score:.0%}  ({sum(c['passed'] for c in result.checks)}/{len(result.checks)} checks)")
    return 0 if result.all_passed else 1


def cmd_validate(args) -> int:
    reports = validate_all(runs=args.runs, only=args.tasks or None)
    if not args.tasks:
        write_reports(reports, args.runs, ROOT / "VALIDATION.md", ROOT / "validation.json")
        print("wrote VALIDATION.md and validation.json")
    return 0 if all(r.valid for r in reports) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forge", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    g = sub.add_parser("grade")
    g.add_argument("task")
    g.add_argument("solution", type=Path)
    g.set_defaults(fn=cmd_grade)
    v = sub.add_parser("validate")
    v.add_argument("--runs", type=int, default=3)
    v.add_argument("tasks", nargs="*")
    v.set_defaults(fn=cmd_validate)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
