"""Kth Largest Element in an Array (LeetCode 215)."""

import random


def _large_case():
    # 10^5 values. Sorting also finishes well under 3 s here, so this checks the answer at the
    # size limit; the intended heap (O(n log k)) or quickselect (O(n) on average) is not timed
    # against a sort.
    rng = random.Random(215)
    nums = [rng.randint(-10**4, 10**4) for _ in range(10**5)]
    k = 1234
    return {"args": [nums, k], "expected": sorted(nums, reverse=True)[k - 1]}


def _random_case(rng):
    # Small values, so duplicates are common and "distinct values" bugs show up
    nums = [rng.randint(-5, 5) for _ in range(rng.randint(1, 8))]
    return {"args": [nums, rng.randint(1, len(nums))]}


TASK = {
    "title": "Kth Largest Element in an Array",
    "title_zh": "数组中的第K个最大元素",
    "difficulty": "Medium",
    "leetcode_id": 215,
    "leetcode_slug": "kth-largest-element-in-an-array",
    "category": "Heap",
    "function_name": "findKthLargest",
    "params": [("nums", "int[]"), ("k", "int")],
    "returns": "int",
    "description_en": "Given an integer array `nums` and an integer `k`, return the `k`-th largest element in the array. Count repeated values separately: in `[3,3,1]`, the 2nd largest is `3` and the 3rd largest is `1`.\n\n**Example 1:** `nums = [7,2,9,4,4], k = 2` → `7` (sorted high to low: 9, 7, 4, 4, 2)\n\n**Example 2:** `nums = [3,3,1], k = 3` → `1`\n\n**Constraints:**\n- $1 \\le \\texttt{k} \\le \\texttt{nums.length} \\le 10^{5}$\n- $-10^{4} \\le \\texttt{nums[i]} \\le 10^{4}$\n\n**Follow-up:** can you do better than sorting the whole array?",
    "description_zh": "给定整数数组 `nums` 和整数 `k`，返回数组中第 `k` 大的元素。重复的值要分开计数：在 `[3,3,1]` 中，第 2 大的是 `3`，第 3 大的是 `1`。\n\n**示例 1：** `nums = [7,2,9,4,4], k = 2` → `7`（从大到小排列为 9, 7, 4, 4, 2）\n\n**示例 2：** `nums = [3,3,1], k = 3` → `1`\n\n**约束：**\n- $1 \\le \\texttt{k} \\le \\texttt{nums.length} \\le 10^{5}$\n- $-10^{4} \\le \\texttt{nums[i]} \\le 10^{4}$\n\n**进阶：** 你能做到比完整排序更快吗？",
    "hint": "Keep a min-heap holding the k largest values seen so far. Its top is the k-th largest once every value has been read.",
    "hint_zh": "维护一个大小为 k 的小根堆，保存目前见过的 k 个最大值。读完所有元素后，堆顶就是第 k 大的值。",
    "cases": [
        {"args": [[7, 2, 9, 4, 4], 2], "expected": 7},
        {"args": [[3, 3, 1], 3], "expected": 1},
        {"args": [[5], 1], "expected": 5},
        # k = n: the smallest value
        {"args": [[1, 2, 3, 4, 5], 5], "expected": 1},
        {"args": [[1, 2, 3, 4, 5], 1], "expected": 5},
        # Negative values, with a duplicate in the top two
        {"args": [[-1, -1, -2, -3], 2], "expected": -1},
        # Every value repeated: counting distinct values instead of occurrences gives the wrong
        # answer here
        {"args": [[8, 8, 8, 8], 4], "expected": 8},
        {"args": [[6, 1, 6, 2, 5, 5], 3], "expected": 5},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''import heapq


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []  # the k largest values seen so far; heap[0] is the smallest of them
        for x in nums:
            heapq.heappush(heap, x)
            if len(heap) > k:
                heapq.heappop(heap)
        return heap[0]''',
    "explanation": "Keep a min-heap of size `k`. Each value is pushed, and when the heap grows past `k`, its smallest element is popped. After all values are read, the heap holds the `k` largest values, and its top (the smallest of them) is the `k`-th largest. Repeated values are separate heap entries, so they are counted correctly.\n\nTime O(n log k), space O(k).",
    "explanation_zh": "维护一个大小为 `k` 的小根堆。每读入一个值就压入堆中，堆超过 `k` 个元素时弹出最小的那个。读完所有值后，堆里正好是最大的 `k` 个值，其中最小的那个（堆顶）就是第 `k` 大的值。重复的值各自占一个堆元素，因此计数正确。\n\n时间 O(n log k)，空间 O(k)。",
}
