"""Search a 2D Matrix (LeetCode 74)."""

import random


def _large_case():
    # A 100 x 100 grid. Scanning all 10^4 cells also finishes well under 3 s, so this checks the
    # answer at the size limit rather than timing an O(m * n) scan against O(log(m * n)).
    rng = random.Random(74)
    flat = sorted(rng.sample(range(-10**4, 10**4 + 1), 10**4))
    matrix = [flat[r * 100:(r + 1) * 100] for r in range(100)]
    if rng.random() < 0.5:
        target = flat[rng.randrange(len(flat))]
    else:
        target = rng.randint(-10**4, 10**4)
    return {"args": [matrix, target], "expected": target in set(flat)}


def _random_case(rng):
    # Non-square grids, so a wrong row/column split (rows where columns belong) shows up
    rows, cols = rng.randint(1, 4), rng.randint(1, 4)
    flat = sorted(rng.sample(range(-20, 21), rows * cols))
    matrix = [flat[r * cols:(r + 1) * cols] for r in range(rows)]
    return {"args": [matrix, rng.randint(-21, 21)]}


GRID = [[1, 4, 6], [8, 10, 13], [15, 20, 21]]

TASK = {
    "title": "Search a 2D Matrix",
    "title_zh": "搜索二维矩阵",
    "difficulty": "Medium",
    "leetcode_id": 74,
    "leetcode_slug": "search-a-2d-matrix",
    "category": "Binary Search",
    "function_name": "searchMatrix",
    "params": [("matrix", "int[][]"), ("target", "int")],
    "returns": "bool",
    "description_en": "You are given an `m x n` integer grid `matrix`. Each row is sorted in ascending order, and every row starts above the last value of the row before it. Read row by row, the whole grid is one sorted sequence. Return `true` if `target` appears in the grid, otherwise `false`.\n\nThe algorithm must run in O(log(m · n)) time.\n\n**Example 1:** `matrix = [[1,4,6],[8,10,13],[15,20,21]], target = 10` → `true`\n\n**Example 2:** `matrix = [[1,4,6],[8,10,13],[15,20,21]], target = 9` → `false`\n\n**Constraints:**\n- `1 <= m, n <= 100`\n- `-10^4 <= matrix[i][j], target <= 10^4`",
    "description_zh": "给定一个 `m x n` 的整数矩阵 `matrix`。每一行都按升序排列，并且每一行的第一个数都大于上一行的最后一个数。按行从左到右读取，整个矩阵就是一个有序序列。如果 `target` 在矩阵中，返回 `true`，否则返回 `false`。\n\n算法的时间复杂度必须是 O(log(m · n))。\n\n**示例 1：** `matrix = [[1,4,6],[8,10,13],[15,20,21]], target = 10` → `true`\n\n**示例 2：** `matrix = [[1,4,6],[8,10,13],[15,20,21]], target = 9` → `false`\n\n**约束：**\n- `1 <= m, n <= 100`\n- `-10^4 <= matrix[i][j], target <= 10^4`",
    "hint": "Treat the grid as one sorted list of m·n numbers. Index `mid` lives at `matrix[mid // n][mid % n]`, so a plain binary search works on it.",
    "hint_zh": "把整个矩阵看成一个长度为 m·n 的有序数组。下标 `mid` 对应 `matrix[mid // n][mid % n]`，直接对它做二分查找即可。",
    "cases": [
        {"args": [GRID, 10], "expected": True},
        {"args": [GRID, 9], "expected": False},
        # Top-left and bottom-right corners: the ends of the flattened range
        {"args": [GRID, 1], "expected": True},
        {"args": [GRID, 21], "expected": True},
        {"args": [GRID, 0], "expected": False},
        {"args": [GRID, 22], "expected": False},
        # Last value of row 0 and first value of row 1: row boundaries
        {"args": [[[1, 4, 6], [8, 10, 13]], 6], "expected": True},
        {"args": [[[1, 4, 6], [8, 10, 13]], 7], "expected": False},
        # Single cell, and a single column (more rows than columns)
        {"args": [[[1]], 2], "expected": False},
        {"args": [[[1], [3], [5]], 3], "expected": True},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        lo, hi = 0, m * n  # half-open range of flat indices
        while lo < hi:
            mid = (lo + hi) // 2
            if matrix[mid // n][mid % n] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo < m * n and matrix[lo // n][lo % n] == target''',
    "explanation": "Because the rows chain together in order, the grid is a sorted list of `m * n` values. Binary search over flat indices `0 .. m*n-1`, converting each index to a cell with `row = index // n` and `col = index % n`. Find the first cell that is at least `target`, then check whether it equals `target`.\n\nTime O(log(m · n)), space O(1).",
    "explanation_zh": "由于各行首尾相接且有序，整个矩阵就是一个包含 `m * n` 个值的有序序列。对平铺后的下标 `0 .. m*n-1` 做二分查找，每个下标用 `行 = 下标 // n`、`列 = 下标 % n` 换算成格子。找到第一个不小于 `target` 的格子，再判断它是否等于 `target`。\n\n时间 O(log(m · n))，空间 O(1)。",
}
