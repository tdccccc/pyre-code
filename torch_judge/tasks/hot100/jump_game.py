"""Jump Game (LeetCode 55)."""


def _can_reach_end(nums: list[int]) -> bool:
    # Flood fill over indices: each index is expanded once, so this is O(n * max jump)
    seen = [False] * len(nums)
    seen[0] = True
    stack = [0]
    while stack:
        i = stack.pop()
        for j in range(i + 1, min(len(nums), i + nums[i] + 1)):
            if not seen[j]:
                seen[j] = True
                stack.append(j)
    return seen[-1]


def _large_case():
    # The answer is false: indices up to n-2 are reachable, but nothing reaches the last index.
    # Each index has one- and two-step routes to it, so there are about Fibonacci(10^4) routes,
    # and a search that tries every route without remembering visited indices never finishes.
    # (An O(n²) DP takes about 1 s here, so it is not caught by time.)
    nums = [2] * 9996 + [2, 1, 0, 0]
    return {"args": [nums], "expected": _can_reach_end(nums)}


def _random_case(rng):
    # Small values make zeros common, so the stuck-at-a-zero cases show up
    return {"args": [[rng.randint(0, 3) for _ in range(rng.randint(1, 8))]]}


TASK = {
    "title": "Jump Game",
    "title_zh": "跳跃游戏",
    "difficulty": "Medium",
    "leetcode_id": 55,
    "leetcode_slug": "jump-game",
    "category": "Greedy",
    "function_name": "canJump",
    "params": [("nums", "int[]")],
    "returns": "bool",
    "description_en": "You start on index `0` of the integer array `nums`. The value at an index is the longest jump you may make from there: from index `i` you may move forward by any whole number of steps from `1` up to `nums[i]`. A value of `0` leaves you stuck.\n\nReturn `true` if you can land on the last index, otherwise `false`.\n\n**Example 1:** `nums = [2,0,2,0,1]` → `true` (0 → 2 → 4)\n\n**Example 2:** `nums = [1,1,0,2]` → `false` (every route ends stuck on index 2)\n\n**Constraints:**\n- `1 <= nums.length <= 10^4`\n- `0 <= nums[i] <= 10^5`",
    "description_zh": "你从整数数组 `nums` 的下标 `0` 出发。下标 `i` 处的数值是从这里能跳的最远距离：可以向前跳 1 到 `nums[i]` 之间的任意整数步。若 `nums[i]` 为 `0`，就会被困在该位置。\n\n判断能否到达最后一个下标：能则返回 `true`，否则返回 `false`。\n\n**示例 1：** `nums = [2,0,2,0,1]` → `true`（0 → 2 → 4）\n\n**示例 2：** `nums = [1,1,0,2]` → `false`（所有路线都会停在下标 2）\n\n**约束：**\n- `1 <= nums.length <= 10^4`\n- `0 <= nums[i] <= 10^5`",
    "hint": "Track the furthest index you can reach so far. Any index beyond that furthest reach is unreachable, so the answer is no once you get past it.",
    "hint_zh": "记录目前能到达的最远下标。超过这个最远位置的下标都无法到达，所以一旦扫描越过它，答案就是否定的。",
    "cases": [
        {"args": [[2, 0, 2, 0, 1]], "expected": True},
        {"args": [[1, 1, 0, 2]], "expected": False},
        {"args": [[0]], "expected": True},
        # Index 1 is unreachable (index 0 is a zero), so its reach of 4 must not count
        {"args": [[0, 3, 3]], "expected": False},
        {"args": [[2, 0, 0]], "expected": True},
        {"args": [[1, 0, 1]], "expected": False},
        # Index 5 is the furthest reachable; the last index is one step past it
        {"args": [[5, 0, 0, 0, 0, 0, 0]], "expected": False},
        {"args": [[3, 2, 1, 0, 0, 1]], "expected": False},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def canJump(self, nums: List[int]) -> bool:
        reach = 0  # furthest index reachable so far
        for i, x in enumerate(nums):
            if i > reach:  # this index can't be reached, so nothing beyond it can be either
                return False
            reach = max(reach, i + x)
        return True''',
    "explanation": "Keep the furthest index reachable so far. Scan left to right: if the current index is beyond that furthest point, it is unreachable and the scan fails. Otherwise, the current index extends the furthest point to `i + nums[i]`. Reaching the last index means it lies within that furthest point.\n\nTime O(n), space O(1).",
    "explanation_zh": "维护到目前为止能到达的最远下标。从左到右扫描：如果当前下标已经超过这个最远位置，说明它无法到达，扫描失败；否则当前下标可以把最远位置推进到 `i + nums[i]`。只要最后一个下标落在最远位置之内，就能到达。\n\n时间 O(n)，空间 O(1)。",
}
