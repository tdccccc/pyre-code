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


# Classic wrong solutions; each must fail somewhere in the hand-written or random cases
MUTANTS = {
    "two_sum": "class Solution:\n    def twoSum(self, nums, target):\n        s = {}\n        for i, x in enumerate(nums):\n            s[x] = i\n            if target - x in s:\n                return [s[target - x], i]\n",
    "group_anagrams": "class Solution:\n    def groupAnagrams(self, strs):\n        g = {}\n        for s in strs:\n            g.setdefault(frozenset(s), []).append(s)\n        return list(g.values())\n",
    "move_zeroes": "class Solution:\n    def moveZeroes(self, nums):\n        i, j = 0, len(nums) - 1\n        while i < j:\n            if nums[i] == 0:\n                nums[i], nums[j] = nums[j], nums[i]\n                j -= 1\n            else:\n                i += 1\n",
    "longest_substring_without_repeating": "class Solution:\n    def lengthOfLongestSubstring(self, s):\n        last, start, best = {}, 0, 0\n        for i, ch in enumerate(s):\n            if ch in last:\n                start = last[ch] + 1\n            last[ch] = i\n            best = max(best, i - start + 1)\n        return best\n",
    "lru_cache": "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, c):\n        self.c, self.d = c, OrderedDict()\n    def get(self, k):\n        return self.d.get(k, -1)\n    def put(self, k, v):\n        if k in self.d:\n            self.d.move_to_end(k)\n        self.d[k] = v\n        if len(self.d) > self.c:\n            self.d.popitem(last=False)\n",
    "binary_tree_level_order": "from collections import deque\nclass Solution:\n    def levelOrder(self, root):\n        if not root:\n            return []\n        res, q = [], deque([root])\n        while q:\n            lvl = []\n            for _ in range(len(q)):\n                n = q.popleft()\n                lvl.append(n.val)\n                if n.right: q.append(n.right)\n                if n.left: q.append(n.left)\n            res.append(lvl)\n        return res\n",
    "coin_change": "class Solution:\n    def coinChange(self, coins, amount):\n        n = 0\n        for c in sorted(coins, reverse=True):\n            n += amount // c\n            amount %= c\n        return n if amount == 0 else -1\n",
}


@pytest.mark.parametrize("task_id", sorted(MUTANTS))
def test_classic_mistake_is_caught_by_random_cases_too(task_id):
    failed = [r["name"] for r in _grade(task_id, MUTANTS[task_id])["results"] if not r["passed"]]
    assert any(name.startswith("Random") for name in failed), failed


def test_random_cases_are_seeded_and_visible():
    task = HOT100_TASKS["two_sum"]
    first = spec.cases_for_mode("two_sum", task, "core")
    again = spec.cases_for_mode("two_sum", task, "core")
    randoms = [c for c in first if c["name"].startswith("Random")]
    assert len(randoms) == spec.RANDOM_CASES and not any(c.get("hidden") for c in randoms)
    assert first == again


def test_hung_case_in_batch_is_pinned_to_that_case():
    # Hangs only on two-element inputs; the batch times out, then per-case reruns find them
    code = (
        "class Solution:\n"
        "    def twoSum(self, nums, target):\n"
        "        while len(nums) == 2:\n"
        "            pass\n"
        "        s = {}\n"
        "        for i, x in enumerate(nums):\n"
        "            if target - x in s:\n"
        "                return [s[target - x], i]\n"
        "            s[x] = i\n"
    )
    cases = spec.cases_for_mode("two_sum", HOT100_TASKS["two_sum"], "core")
    results = _grade("two_sum", code)["results"]
    for case, result in zip(cases, results):
        two_elements = not case.get("hidden") and len(case["args"][0]) == 2
        assert result["passed"] != two_elements, case["name"]


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
