"""Daily Temperatures (LeetCode 739)."""

import random


def _waits(temps: list) -> list:
    # Jump pointers: out[j] is the wait from day j, so the search for a warmer day after i can
    # skip over the whole stretch that day j already skipped
    n = len(temps)
    out = [0] * n
    for i in range(n - 2, -1, -1):
        j = i + 1
        while j < n and temps[j] <= temps[i]:
            if out[j] == 0:
                j = n  # nothing after day j is warmer than temps[j], so nothing is warmer than temps[i]
            else:
                j += out[j]
        out[i] = j - i if j < n else 0
    return out


def _large_case():
    # 10^5 days, the constraint limit. 90000 days sit at the same cold temperature, so a naive
    # scan from each of them runs to the end of that stretch: about 4·10^9 steps
    rng = random.Random(739)
    temps = [30] * 90000 + [rng.randint(30, 100) for _ in range(10000)]
    return {"args": [temps], "expected": _waits(temps)}


def _random_case(rng):
    return {"args": [[rng.randint(30, 40) for _ in range(rng.randint(1, 8))]]}


TASK = {
    "title": "Daily Temperatures",
    "title_zh": "每日温度",
    "difficulty": "Medium",
    "leetcode_id": 739,
    "leetcode_slug": "daily-temperatures",
    "category": "Stack",
    "function_name": "dailyTemperatures",
    "params": [("temperatures", "int[]")],
    "returns": "int[]",
    "description_en": "You are given the daily temperatures `temperatures`. For each day, return how many days you must wait until a strictly warmer day comes. If no warmer day follows, the answer for that day is `0`.\n\n**Example 1:** `temperatures = [60,50,55,70,40]` → `[3,1,1,0,0]`\n\n**Example 2:** `temperatures = [35,30,40,20]` → `[2,1,0,0]`\n\n**Constraints:**\n- `1 <= temperatures.length <= 10^5`\n- `30 <= temperatures[i] <= 100`",
    "description_zh": "给定每日温度数组 `temperatures`。对每一天，返回还需要等待多少天才会出现一个严格更高的温度。如果之后不存在更高的温度，该天的答案为 `0`。\n\n**示例 1：** `temperatures = [60,50,55,70,40]` → `[3,1,1,0,0]`\n\n**示例 2：** `temperatures = [35,30,40,20]` → `[2,1,0,0]`\n\n**约束：**\n- `1 <= temperatures.length <= 10^5`\n- `30 <= temperatures[i] <= 100`",
    "hint": "Keep a stack of days still waiting for a warmer day, and their temperatures decreasing from bottom to top. When a day is warmer than the top of the stack, it is the answer for every waiting day it beats.",
    "hint_zh": "用一个栈保存仍在等待更高温度的日期，栈中温度从栈底到栈顶递减。当某天比栈顶温度更高时，它就是栈中所有比它矮的等待日期的答案。",
    "cases": [
        {"args": [[60, 50, 55, 70, 40]], "expected": [3, 1, 1, 0, 0]},
        {"args": [[35, 30, 40, 20]], "expected": [2, 1, 0, 0]},
        {"args": [[30]], "expected": [0]},
        # Equal temperatures are not warmer: a "<=" comparison counts them and gets this wrong
        {"args": [[30, 30, 30]], "expected": [0, 0, 0]},
        {"args": [[31, 32, 33, 34]], "expected": [1, 1, 1, 0]},
        {"args": [[34, 33, 32, 31]], "expected": [0, 0, 0, 0]},
        # Repeated 70s waiting for a warmer day, several answers at once
        {"args": [[70, 71, 70, 70, 72]], "expected": [1, 3, 2, 1, 0]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        answer = [0] * len(temperatures)
        stack = []  # days still waiting for a warmer day, coldest on top
        for day, temp in enumerate(temperatures):
            while stack and temperatures[stack[-1]] < temp:
                prev = stack.pop()
                answer[prev] = day - prev
            stack.append(day)
        return answer''',
    "explanation": "Monotonic stack of day indices whose temperatures are still waiting for a warmer day, with temperatures non-increasing from bottom to top. When day `d` arrives, every waiting day whose temperature is below `temperatures[d]` has found its answer, `d - prev`. Those days are popped, and `d` is pushed. Each day is pushed and popped at most once.\n\nTime O(n), space O(n).",
    "explanation_zh": "用单调栈保存仍在等待更高温度的日期下标，栈中温度从栈底到栈顶不增。当第 `d` 天到来时，所有温度低于 `temperatures[d]` 的等待日期都找到了答案 `d - prev`。把它们弹出，再把 `d` 入栈。每一天最多入栈和出栈一次。\n\n时间 O(n)，空间 O(n)。",
}
