"""Check registry used by task graders.

A grader is a plain Python module. Each function decorated with @check is one
hidden test; it receives the solution module and passes by returning, fails by
raising AssertionError. Checks run in definition order.
"""

from __future__ import annotations

from types import ModuleType
from typing import Callable

CheckFn = Callable[[ModuleType], None]


def check(name: str, weight: int = 1, kind: str = "correctness") -> Callable[[CheckFn], CheckFn]:
    if kind not in ("correctness", "performance", "robustness"):
        raise ValueError(f"unknown check kind {kind!r}")

    def deco(fn: CheckFn) -> CheckFn:
        fn._forge_check = {"name": name, "weight": weight, "kind": kind}  # type: ignore[attr-defined]
        return fn

    return deco


def collect(module: ModuleType) -> list[tuple[dict, CheckFn]]:
    return [(fn._forge_check, fn) for fn in vars(module).values()
            if callable(fn) and hasattr(fn, "_forge_check")]
