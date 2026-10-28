"""Climbing Stairs (LeetCode 70)."""

import random


def _ways(n: int) -> int:
    # Iterate upward from step 0 and step 1, which each have one way to be reached
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def _large_case():
    # n = 45 is the maximum. The plain recursion f(n) = f(n-1) + f(n-2) makes about 3·10^9 calls
    # here, so it times out; the iterative version is instant
    return {"args": [45], "expected": _ways(45)}


def _random_case(rng):
    return {"args": [rng.randint(1, 12)]}


TASK = {
    "title": "Climbing Stairs",
    "title_zh": "爬楼梯",
    "difficulty": "Easy",
    "leetcode_id": 70,
    "leetcode_slug": "climbing-stairs",
    "category": "Dynamic Programming",
    "function_name": "climbStairs",
    "params": [("n", "int")],
    "returns": "int",
    "description_en": "A staircase has `n` steps, and you start on the ground below step 1. Each move covers either one step or two. Count the distinct sequences of moves that take you to the top, where the order of moves matters.\n\n**Example 1:** `n = 3` → `3` (1+1+1, 1+2, 2+1)\n\n**Example 2:** `n = 6` → `13`\n\n**Constraints:**\n- `1 <= n <= 45`",
    "description_zh": "一段楼梯共有 `n` 级台阶，你从第 1 级台阶之下出发。每一步可以迈 1 级或 2 级。统计能登顶的不同走法有多少种，走法的先后顺序不同即视为不同。\n\n**示例 1：** `n = 3` → `3`（1+1+1、1+2、2+1）\n\n**示例 2：** `n = 6` → `13`\n\n**约束：**\n- `1 <= n <= 45`",
    "hint": "The last move is either a one-step or a two-step, so ways(n) = ways(n - 1) + ways(n - 2). Only the last two values are needed.",
    "hint_zh": "最后一步要么迈 1 级，要么迈 2 级，所以 ways(n) = ways(n - 1) + ways(n - 2)。只需保存最近两个值。",
    "cases": [
        {"args": [3], "expected": 3},
        {"args": [6], "expected": 13},
        {"args": [1], "expected": 1},
        # Returning the plain Fibonacci number F(n) gives 1 here, but there are two ways (1+1, 2)
        {"args": [2], "expected": 2},
        {"args": [10], "expected": 89},
        {"args": [20], "expected": 10946},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def climbStairs(self, n: int) -> int:
        prev, cur = 1, 1  # ways to reach step 0 and step 1
        for _ in range(n - 1):
            prev, cur = cur, prev + cur  # the next step is reached from either of the last two
        return cur''',
    "explanation": "Let `ways(k)` be the number of ways to reach step `k`. The final move is one step from `k - 1` or two steps from `k - 2`, so `ways(k) = ways(k - 1) + ways(k - 2)`, with one way to reach each of steps 0 and 1. This is the Fibonacci recurrence. Keeping only the last two values gives the answer in a single loop.\n\nTime O(n), space O(1).",
    "explanation_zh": "设 `ways(k)` 为到达第 `k` 级台阶的走法数。最后一步要么从第 `k - 1` 级迈 1 级上来，要么从第 `k - 2` 级迈 2 级上来，所以 `ways(k) = ways(k - 1) + ways(k - 2)`，并且第 0 级和第 1 级各只有一种走法。这就是斐波那契递推。只保存最近两个值，就能在一个循环内算出答案。\n\n时间 O(n)，空间 O(1)。",
}
