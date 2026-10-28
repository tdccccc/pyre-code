"""Tests for the Hot 100 judge: every reference solution passes in both modes, and the
grader rejects wrong, slow and malformed answers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_judge.algo import judge, runner, spec
from torch_judge.tasks import HOT100_TASKS


def _grade(task_id, code, mode="core", indices=None):
    return judge.grade(task_id, HOT100_TASKS[task_id], code, mode, indices)


@pytest.mark.parametrize("mode", spec.MODES)
@pytest.mark.parametrize("task_id", sorted(HOT100_TASKS))
def test_reference_solution_passes(task_id, mode):
    task = HOT100_TASKS[task_id]
    code = task["solution"] if mode == "core" else spec.acm_solution(task)
    result = _grade(task_id, code, mode)
    failures = [(r["name"], r["error"]) for r in result["results"] if not r["passed"]]
    assert result["allPassed"], failures


@pytest.mark.parametrize("task_id", sorted(HOT100_TASKS))
def test_task_has_two_visible_samples_and_a_large_case(task_id):
    for mode in spec.MODES:
        cases = spec.cases_for_mode(task_id, HOT100_TASKS[task_id], mode)
        assert len(cases) >= 3
        assert not any(c.get("hidden") for c in cases[:2])


def test_brute_force_times_out_on_large_input():
    brute = (
        "class Solution:\n"
        "    def twoSum(self, nums, target):\n"
        "        for i in range(len(nums)):\n"
        "            for j in range(i + 1, len(nums)):\n"
        "                if nums[i] + nums[j] == target:\n"
        "                    return [i, j]\n"
    )
    result = _grade("two_sum", brute)
    verdicts = {r["name"]: r for r in result["results"]}
    assert all(r["passed"] for name, r in verdicts.items() if name != "Large input")
    assert "Time Limit Exceeded" in verdicts["Large input"]["error"]


def test_wrong_answer_reports_actual_value():
    result = _grade("coin_change", "class Solution:\n    def coinChange(self, coins, amount):\n        return 7\n", indices=[0])
    case = result["results"][0]
    assert not case["passed"] and case["error"] == "Wrong Answer"
    assert case["actual"] == "7" and case["expected"] == "2"


def test_inplace_problem_requires_mutation():
    returns_copy = "class Solution:\n    def moveZeroes(self, nums):\n        return sorted(nums, key=lambda x: x == 0)\n"
    assert not _grade("move_zeroes", returns_copy, indices=[0])["allPassed"]


def test_runtime_error_points_at_learner_line():
    result = _grade("coin_change", "class Solution:\n    def coinChange(self, coins, amount):\n        return coins[99]\n", indices=[0])
    assert result["results"][0]["error"] == "IndexError: list index out of range (line 3)"


def test_acm_accepts_any_order_for_unordered_answers():
    assert _grade("two_sum", "print(3, 2)\n", "acm", [0])["allPassed"]


def test_acm_rejects_malformed_output():
    case = _grade("two_sum", "print('[2, 3]')\n", "acm", [0])["results"][0]
    assert not case["passed"] and case["error"].startswith("Wrong Answer (output format")


def test_acm_runtime_error_hides_temp_path():
    case = _grade("two_sum", "print(1 / 0)\n", "acm", [0])["results"][0]
    assert "ZeroDivisionError" in case["error"] and 'File "solution.py"' in case["error"]


@pytest.mark.parametrize("type_name, value", [
    ("int[]", [3, -1, 0]),
    ("int[]", []),
    ("int[][]", [[5], [3, 8], [4, 7, 9]]),
    ("TreeNode", [5, 3, 8, None, 4]),
    ("bool", True),
    ("str[]", ["ab", "c"]),
])
def test_acm_output_round_trips(type_name, value):
    assert runner.parse_acm_output(type_name, runner.acm_output_text(type_name, value) + "\n") == value


def test_tree_codec_round_trips():
    values = [1, 2, 3, None, 4, None, 5, 6]
    assert runner.tree_to_array(runner.build_tree(values)) == values
