"""Sliding Window Maximum (LeetCode 239)."""

import heapq
import random


def _window_max(nums: list[int], k: int) -> list[int]:
    # Independent method: a max-heap with lazy deletion of indices that left the window
    heap, out = [], []
    for i, x in enumerate(nums):
        heapq.heappush(heap, (-x, i))
        if i >= k - 1:
            while heap[0][1] <= i - k:
                heapq.heappop(heap)
            out.append(-heap[0][0])
    return out


def _large_case():
    # 10^5 values with a window of 25000. Taking max() of every window costs about 2·10⁹
    # comparisons; the monotonic deque does one pass.
    rng = random.Random(239)
    nums = [rng.randint(-10**4, 10**4) for _ in range(100000)]
    k = 25000
    return {"args": [nums, k], "expected": _window_max(nums, k)}


def _random_case(rng):
    n = rng.randint(1, 10)
    return {"args": [[rng.randint(-5, 5) for _ in range(n)], rng.randint(1, n)]}


TASK = {
    "title": "Sliding Window Maximum",
    "title_zh": "滑动窗口最大值",
    "difficulty": "Hard",
    "leetcode_id": 239,
    "leetcode_slug": "sliding-window-maximum",
    "category": "Substring",
    "function_name": "maxSlidingWindow",
    "params": [("nums", "int[]"), ("k", "int")],
    "returns": "int[]",
    "description_en": "Given an integer array `nums` and an integer `k`, a window of `k` consecutive values slides from the left end of the array to the right end, one step at a time. Return the largest value inside the window at each position, in order.\n\n**Example 1:** `nums = [4,2,12,11,-5,7], k = 3` → `[12,12,12,11]`\n\n**Example 2:** `nums = [9,8,7,6], k = 4` → `[9]`\n\n**Constraints:**\n- $1 \\le \\texttt{k} \\le \\texttt{nums.length} \\le 10^{5}$\n- $-10^{4} \\le \\texttt{nums[i]} \\le 10^{4}$",
    "description_zh": "给定整数数组 `nums` 和整数 `k`。一个包含 `k` 个连续元素的窗口从数组最左端开始，每次向右移动一步，直到最右端。依次返回每个窗口位置中的最大值。\n\n**示例 1：** `nums = [4,2,12,11,-5,7], k = 3` → `[12,12,12,11]`\n\n**示例 2：** `nums = [9,8,7,6], k = 4` → `[9]`\n\n**约束：**\n- $1 \\le \\texttt{k} \\le \\texttt{nums.length} \\le 10^{5}$\n- $-10^{4} \\le \\texttt{nums[i]} \\le 10^{4}$",
    "hint": "Keep a deque of indices whose values decrease from front to back. Its front is the window's maximum. Drop smaller values from the back when a new value arrives, and drop the front once its index leaves the window.",
    "hint_zh": "维护一个下标双端队列，队列中的值从前到后递减。队首就是窗口的最大值。新值到来时，弹出队尾比它小的下标；队首下标离开窗口时，从队首弹出。",
    "cases": [
        {"args": [[4, 2, 12, 11, -5, 7], 3], "expected": [12, 12, 12, 11]},
        {"args": [[9, 8, 7, 6], 4], "expected": [9]},
        {"args": [[1], 1], "expected": [1]},
        {"args": [[1, -1], 1], "expected": [1, -1]},
        # The maximum must leave the window once its index falls out; a deque that never drops
        # the front returns 9 for every window here
        {"args": [[9, 1, 2, 3], 2], "expected": [9, 2, 3]},
        {"args": [[7, 7, 7, 7], 2], "expected": [7, 7, 7]},
        {"args": [[-7, -8, -9, -10], 2], "expected": [-7, -8, -9]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''from collections import deque


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        dq = deque()  # indices; their values decrease from front to back
        out = []
        for i, x in enumerate(nums):
            if dq and dq[0] <= i - k:
                dq.popleft()  # the front index has left the window
            while dq and nums[dq[-1]] <= x:
                dq.pop()  # a smaller value before a bigger one can never be a maximum
            dq.append(i)
            if i >= k - 1:
                out.append(nums[dq[0]])
        return out''',
    "explanation": "A monotonic deque of indices. Values are kept decreasing from front to back, so the front is the largest value in the current window. A new value removes every smaller value behind it, because those can never be a window maximum again. The front is removed once its index is `k` or more positions behind. Each index enters and leaves the deque at most once.\n\nTime O(n), space O(k).",
    "explanation_zh": "单调双端队列，保存下标，对应的值从前到后递减，所以队首就是当前窗口的最大值。新元素到来时，队尾比它小的元素之后不可能再成为窗口最大值，直接弹出。队首下标落后当前位置 `k` 或更多时，它已经离开窗口，也弹出。每个下标最多入队和出队一次。\n\n时间 O(n)，空间 O(k)。",
}
