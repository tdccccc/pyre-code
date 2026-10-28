"""Grade a Hot 100 submission in core or ACM mode.

Every case runs in its own ``python -I`` subprocess with a wall-clock limit, so an
infinite loop or a crash costs one case instead of the grading service.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from torch_judge.algo import runner, spec

TIME_LIMIT_S = 3.0
RUNNER_PATH = Path(runner.__file__)
MAX_STDERR_LINES = 12


def _result(case: dict, task: dict, mode: str, **fields) -> dict:
    hidden = case.get("hidden", False)
    return {
        "name": case["name"],
        "input": None if hidden else spec.display_input(task, case, mode),
        "expected": None if hidden else spec.display_expected(task, case, mode),
        **fields,
    }


def _run_core_case(code: str, task: dict, case: dict) -> dict:
    job = {"code": code, "spec": spec.runner_spec(task), "case": {k: case[k] for k in ("args", "ops", "expected") if k in case}}
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(RUNNER_PATH)],
            input=json.dumps(job), capture_output=True, text=True, timeout=TIME_LIMIT_S,
        )
    except subprocess.TimeoutExpired:
        return _result(case, task, "core", passed=False, execTimeMs=TIME_LIMIT_S * 1000,
                       error=f"Time Limit Exceeded (> {TIME_LIMIT_S:g}s)")
    wall_ms = (time.perf_counter() - start) * 1000

    lines = proc.stdout.rstrip("\n").split("\n")
    try:
        outcome = json.loads(lines[-1])
    except (json.JSONDecodeError, IndexError):
        stderr = _tail(proc.stderr) or "the program exited unexpectedly"
        return _result(case, task, "core", passed=False, execTimeMs=wall_ms, error=f"Runtime Error\n{stderr}")

    output = outcome.get("stdout") or None
    elapsed = outcome.get("timeMs", wall_ms)
    if outcome["status"] == "error":
        return _result(case, task, "core", passed=False, execTimeMs=elapsed, error=outcome["error"], output=output)
    passed = outcome["status"] == "ok"
    return _result(case, task, "core", passed=passed, execTimeMs=elapsed, actual=outcome["actual"],
                   error=None if passed else "Wrong Answer", output=output)


def _tail(stderr: str, path: str | None = None) -> str:
    if path:
        stderr = stderr.replace(path, "solution.py")
    return "\n".join(stderr.strip().split("\n")[-MAX_STDERR_LINES:])


def _run_acm_case(program: Path, task: dict, case: dict) -> dict:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(program)],
            input=spec.acm_stdin(task, case), capture_output=True, text=True,
            timeout=TIME_LIMIT_S, cwd=program.parent,
        )
    except subprocess.TimeoutExpired:
        return _result(case, task, "acm", passed=False, execTimeMs=TIME_LIMIT_S * 1000,
                       error=f"Time Limit Exceeded (> {TIME_LIMIT_S:g}s)")
    elapsed = (time.perf_counter() - start) * 1000
    actual = runner._clip(proc.stdout.rstrip("\n"))

    if proc.returncode != 0:
        return _result(case, task, "acm", passed=False, execTimeMs=elapsed, actual=actual or None,
                       error=f"Runtime Error\n{_tail(proc.stderr, str(program))}")

    try:
        passed = _acm_matches(task, case, proc.stdout)
        error = None if passed else "Wrong Answer"
    except (ValueError, NotImplementedError) as exc:
        passed, error = False, f"Wrong Answer (output format: {exc})"
    return _result(case, task, "acm", passed=passed, execTimeMs=elapsed, actual=actual, error=error)


def _acm_matches(task: dict, case: dict, stdout: str) -> bool:
    if task.get("design"):
        got = [line.strip() for line in stdout.strip().split("\n")] if stdout.strip() else []
        want = spec.acm_expected_stdout(task, case)
        return got == (want.split("\n") if want else [])
    parsed = runner.parse_acm_output(spec.acm_output_type(task), stdout)
    return runner.check(task.get("check", "exact"), parsed, case["expected"])


def grade(task_id: str, task: dict, code: str, mode: str = "core", test_indices: list[int] | None = None) -> dict:
    if mode not in spec.MODES:
        return {"passed": 0, "total": 0, "allPassed": False, "results": [], "totalTimeMs": 0.0, "error": f"Unknown mode '{mode}'"}

    cases = spec.cases_for_mode(task_id, task, mode)
    if test_indices is not None:
        cases = [cases[i] for i in test_indices if 0 <= i < len(cases)]

    if mode == "acm":
        with tempfile.TemporaryDirectory(prefix="pyre-acm-") as tmp:
            program = Path(tmp) / "solution.py"
            program.write_text(code, encoding="utf-8")
            results = [_run_acm_case(program, task, case) for case in cases]
    else:
        results = [_run_core_case(code, task, case) for case in cases]

    passed = sum(r["passed"] for r in results)
    return {
        "passed": passed,
        "total": len(results),
        "allPassed": passed == len(results),
        "results": results,
        "totalTimeMs": sum(r["execTimeMs"] for r in results),
    }
