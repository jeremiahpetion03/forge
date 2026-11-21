"""Reports must describe the task files actually in the repository."""

import json
import shutil

from forge.__main__ import main
from forge.manifest import build, stale_tasks, task_sha256
from forge.runner import ROOT, TASKS_DIR
from forge.validate import list_tasks

TASK = TASKS_DIR / "03_gradient_accumulation"


def copy_task(tmp_path):
    dst = tmp_path / TASK.name
    shutil.copytree(TASK, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def test_hash_is_deterministic():
    assert task_sha256(TASK) == task_sha256(TASK)


def test_hash_changes_when_any_task_file_changes(tmp_path):
    t = copy_task(tmp_path)
    before = task_sha256(t)
    grader = t / "grader.py"
    grader.write_text(grader.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert task_sha256(t) != before


def test_adding_a_mutant_changes_the_hash(tmp_path):
    t = copy_task(tmp_path)
    before = task_sha256(t)
    (t / "mutants" / "new.py").write_text("x = 1\n", encoding="utf-8")
    assert task_sha256(t) != before


def test_crlf_checkout_hashes_like_lf(tmp_path):
    # Windows git checkouts may convert line endings; the report must not look stale.
    t = copy_task(tmp_path)
    before = task_sha256(t)
    for p in t.rglob("*.py"):
        p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    assert task_sha256(t) == before


def test_stale_detects_edited_added_and_removed_tasks(tmp_path):
    t = copy_task(tmp_path)
    report = {"manifest": build([t], runs=1, root=tmp_path)}
    assert stale_tasks(report, [t]) == []
    (t / "README.md").write_text("changed spec\n", encoding="utf-8")
    assert stale_tasks(report, [t]) == [t.name]
    assert stale_tasks({"manifest": {"task_sha256": {}}}, [TASK]) == [TASK.name]   # added
    assert stale_tasks(report, []) == [t.name]                                      # removed


def test_check_report_cli_exit_codes(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"manifest": build(list_tasks(), runs=1, root=ROOT)}), encoding="utf-8")
    assert main(["check-report", "--report", str(good)]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"manifest": {"task_sha256": {}}}), encoding="utf-8")
    assert main(["check-report", "--report", str(bad)]) == 1
    assert main(["check-report", "--report", str(tmp_path / "missing.json")]) == 1


def test_committed_report_matches_current_tasks():
    # Fails whenever a task is edited without re-running validation, so a
    # stale VALIDATION.md can never be committed unnoticed.
    report = json.loads((ROOT / "validation.json").read_text(encoding="utf-8"))
    assert stale_tasks(report, list_tasks()) == [], "run: python -m forge validate --runs 10"
