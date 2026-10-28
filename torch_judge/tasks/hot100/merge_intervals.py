"""Merge Intervals (LeetCode 56)."""

import random


def _merge(intervals):
    # Sort by start, then extend the current run whenever the next interval starts before it ends
    ordered = sorted(intervals, key=lambda iv: iv[0])
    out, cur_s, cur_e = [], None, None
    for s, e in ordered:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            out.append([cur_s, cur_e])
            cur_s, cur_e = s, e
    if cur_s is not None:
        out.append([cur_s, cur_e])
    return out


def _large_case():
    # 10^4 short intervals that mostly overlap their neighbours, so the merges form long chains
    rng = random.Random(56)
    intervals = []
    for _ in range(10000):
        start = rng.randint(0, 9970)
        intervals.append([start, start + rng.randint(0, 30)])
    return {"args": [intervals], "expected": _merge(intervals)}


def _random_case(rng):
    intervals = []
    for _ in range(rng.randint(1, 6)):
        start = rng.randint(0, 10)
        intervals.append([start, start + rng.randint(0, 4)])
    return {"args": [intervals]}


TASK = {
    "title": "Merge Intervals",
    "title_zh": "合并区间",
    "difficulty": "Medium",
    "leetcode_id": 56,
    "leetcode_slug": "merge-intervals",
    "category": "Arrays",
    "function_name": "merge",
    "params": [("intervals", "int[][]")],
    "returns": "int[][]",
    "description_en": "You are given a list `intervals`, where `intervals[i] = [start, end]` is a closed range. Merge every group of ranges that overlap, and return the result as a list of disjoint ranges sorted by start. Two ranges that only touch at an endpoint count as overlapping, so `[1,4]` and `[4,5]` merge into `[1,5]`.\n\n**Example 1:** `intervals = [[6,9],[1,2],[4,6],[10,12]]` → `[[1,2],[4,9],[10,12]]`\n\n**Example 2:** `intervals = [[3,3],[0,1]]` → `[[0,1],[3,3]]`\n\n**Constraints:**\n- `1 <= intervals.length <= 10^4`\n- `intervals[i].length == 2`\n- `0 <= start <= end <= 10^4`",
    "description_zh": "给定区间列表 `intervals`，其中 `intervals[i] = [start, end]` 表示一个闭区间。把所有有重叠的区间合并起来，返回合并后互不重叠的区间列表，并按起点升序排列。只在端点处相接的两个区间也算重叠，所以 `[1,4]` 和 `[4,5]` 应合并为 `[1,5]`。\n\n**示例 1：** `intervals = [[6,9],[1,2],[4,6],[10,12]]` → `[[1,2],[4,9],[10,12]]`\n\n**示例 2：** `intervals = [[3,3],[0,1]]` → `[[0,1],[3,3]]`\n\n**约束：**\n- `1 <= intervals.length <= 10^4`\n- `intervals[i].length == 2`\n- `0 <= start <= end <= 10^4`",
    "hint": "Sort by start. After that, comparing each interval only with the last merged one is enough: when they overlap, the merged end is the larger of the two ends.",
    "hint_zh": "先按起点排序。之后只需把每个区间和最后一个合并结果比较：有重叠时，合并后的结尾取两者中较大的那个。",
    "cases": [
        {"args": [[[6, 9], [1, 2], [4, 6], [10, 12]]], "expected": [[1, 2], [4, 9], [10, 12]]},
        {"args": [[[3, 3], [0, 1]]], "expected": [[0, 1], [3, 3]]},
        # Touching endpoints merge
        {"args": [[[1, 4], [4, 5]]], "expected": [[1, 5]]},
        # A contained interval must not shorten the merged end
        {"args": [[[1, 10], [2, 3], [4, 5]]], "expected": [[1, 10]]},
        {"args": [[[1, 4], [0, 0]]], "expected": [[0, 0], [1, 4]]},
        # Gaps stay gaps
        {"args": [[[2, 3], [4, 5], [6, 7]]], "expected": [[2, 3], [4, 5], [6, 7]]},
        {"args": [[[5, 7]]], "expected": [[5, 7]]},
        # Input is not sorted
        {"args": [[[8, 10], [1, 3], [2, 6]]], "expected": [[1, 6], [8, 10]]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return merged''',
    "explanation": "Sort by start, then sweep while keeping the last merged interval. If the next interval starts at or before that interval's end, it overlaps (touching counts), so extend the end to the larger of the two. Otherwise the last interval is finished and the next one starts a new output. Taking `max` keeps a contained interval from shortening the result.\n\nTime O(n log n), space O(n) for the output.",
    "explanation_zh": "先按起点排序，然后一边扫描一边保留最后一个合并结果。如果下一个区间的起点不晚于它的终点，两者就重叠（端点相接也算），于是把终点更新为两者中较大的那个；否则上一个合并结果已经完成，下一个区间开启新的结果。取 `max` 可以保证被包含的区间不会缩短结果。\n\n时间 O(n log n)，输出之外的空间 O(n)。",
}
