"""The harness must survive anything a submission does, and must refuse to
certify a task whose checks don't work."""

import json
import textwrap

import pytest

from forge.__main__ import main
from forge.runner import TASKS_DIR, run
from forge.validate import list_tasks, validate_all, validate_task

TASK = TASKS_DIR / "03_gradient_accumulation"


def write(path, code):
    path.write_text(textwrap.dedent(code), encoding="utf-8")
    return path


def test_reference_passes_and_scores_100(tmp_path):
    r = run(TASK, TASK / "reference.py")
    assert r.status == "ok" and r.all_passed and r.score == 1.0


def test_hanging_solution_times_out(tmp_path):
    sol = write(tmp_path / "hang.py", "while True:\n    pass\n")
    r = run(TASK, sol, timeout=2)
    assert r.status == "timeout" and r.score == 0.0


def test_syntax_error_is_reported_not_raised(tmp_path):
    sol = write(tmp_path / "bad.py", "def accumulated_gradient(:\n")
    r = run(TASK, sol)
    assert r.status == "import_error" and "SyntaxError" in r.error


def test_hard_crash_is_contained(tmp_path):
    sol = write(tmp_path / "crash.py", "import os\nos._exit(3)\n")
    r = run(TASK, sol)
    assert r.status == "crashed" and "exit code 3" in r.error


def test_missing_function_fails_checks_with_a_clear_message(tmp_path):
    sol = write(tmp_path / "empty.py", "x = 1\n")
    r = run(TASK, sol)
    assert r.status == "ok" and r.score == 0.0
    assert all("AttributeError" in c["message"] for c in r.checks)


def test_solution_output_does_not_confuse_the_parser(tmp_path):
    ref = (TASK / "reference.py").read_text(encoding="utf-8")
    sol = write(tmp_path / "chatty.py", 'print("hello\\nFORGE noise {not json}")\n' + ref)
    assert run(TASK, sol).all_passed


def test_solution_cannot_leak_state_between_runs(tmp_path):
    # Each run is a fresh interpreter, so a module-level counter restarts.
    sol = write(tmp_path / "stateful.py", """
        import os, numpy as np
        flag = os.path.join(os.getcwd(), "ran")
        assert not os.path.exists(flag), "scratch directory was reused"
        open(flag, "w").close()
        def accumulated_gradient(batches, w, grad_fn):
            return np.zeros_like(w)
    """)
    assert run(TASK, sol).status == "ok"
    assert run(TASK, sol).status == "ok"


def test_cli_exit_codes(tmp_path, capsys):
    assert main(["grade", str(TASK), str(TASK / "reference.py")]) == 0
    assert main(["grade", str(TASK), str(TASK / "starter.py")]) == 1
    assert "score" in capsys.readouterr().out


def _make_task(root, reference, mutant):
    t = root / "toy"
    (t / "mutants").mkdir(parents=True)
    write(t / "task.json", json.dumps({"title": "toy", "difficulty": "easy", "time_limit_s": 20}))
    write(t / "grader.py", """
        from forge.checks import check

        @check("doubles")
        def doubles(sol):
            assert sol.f(2) == 4 and sol.f(-3) == -6
    """)
    write(t / "starter.py", "def f(x):\n    return x\n")
    write(t / "reference.py", reference)
    write(t / "mutants" / "m.py", mutant)
    return t


def test_validator_accepts_a_good_task(tmp_path):
    t = _make_task(tmp_path, "def f(x):\n    return 2 * x\n", "def f(x):\n    return x + 2\n")
    assert validate_task(t, runs=2).valid


def test_validator_rejects_a_task_whose_reference_fails(tmp_path):
    t = _make_task(tmp_path, "def f(x):\n    return x * x\n", "def f(x):\n    return x + 2\n")
    rep = validate_task(t, runs=2)
    assert not rep.valid and not rep.reference_passes


def test_validator_rejects_a_task_that_misses_a_mutant(tmp_path):
    # x + x is a different program but equivalent behavior: the grader can't
    # tell it apart, so this "mutant" escapes and the task must be flagged.
    t = _make_task(tmp_path, "def f(x):\n    return 2 * x\n", "def f(x):\n    return x + x\n")
    rep = validate_task(t, runs=2)
    assert not rep.valid and any("passes every check" in p for p in rep.problems)


def test_validator_rejects_a_flaky_task(tmp_path):
    t = _make_task(tmp_path, "def f(x):\n    return 2 * x\n", "def f(x):\n    return x + 2\n")
    write(t / "grader.py", """
        import random
        from forge.checks import check

        @check("coin flip")
        def flaky(sol):
            assert sol.f(2) == 4
            assert random.random() < 0.5
    """)
    rep = validate_task(t, runs=20)  # P(20 identical coin flips) = 2 / 2**20
    assert not rep.not_flaky and not rep.valid


@pytest.mark.parametrize("task", [p.name for p in list_tasks()])
def test_every_shipped_task_is_valid(task):
    (rep,) = validate_all(runs=2, only=[task], verbose=False)
    assert rep.valid, rep.problems
