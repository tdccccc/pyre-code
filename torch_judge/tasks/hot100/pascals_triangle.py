"""Pascal's Triangle (LeetCode 118)."""

import math


def _large_case():
    # The maximum of 30 rows. Every entry comes from the binomial formula C(row, k), an
    # independent route from the row-by-row construction the reference uses. Any reasonable
    # method finishes in milliseconds at this size, so the time limit can't be exceeded here.
    return {"args": [30], "expected": [[math.comb(r, k) for k in range(r + 1)] for r in range(30)]}


def _random_case(rng):
    return {"args": [rng.randint(1, 8)]}


TASK = {
    "title": "Pascal's Triangle",
    "title_zh": "杨辉三角",
    "difficulty": "Easy",
    "leetcode_id": 118,
    "leetcode_slug": "pascals-triangle",
    "category": "Dynamic Programming",
    "function_name": "generate",
    "params": [("numRows", "int")],
    "returns": "int[][]",
    "description_en": "Build the first `numRows` rows of Pascal's triangle. The top row is `[1]`. Each row starts and ends with `1`, and every other entry equals the sum of the two entries directly above it in the previous row. Return the rows as a list of lists, top row first.\n\n**Example 1:** `numRows = 4` → `[[1],[1,1],[1,2,1],[1,3,3,1]]`\n\n**Example 2:** `numRows = 1` → `[[1]]`\n\n**Constraints:**\n- `1 <= numRows <= 30`",
    "description_zh": "生成杨辉三角的前 `numRows` 行。第一行是 `[1]`。每一行的首尾都是 `1`，其余每个数等于上一行中它正上方两个数之和。把各行按从上到下的顺序以列表的列表形式返回。\n\n**示例 1：** `numRows = 4` → `[[1],[1,1],[1,2,1],[1,3,3,1]]`\n\n**示例 2：** `numRows = 1` → `[[1]]`\n\n**约束：**\n- `1 <= numRows <= 30`",
    "hint": "Build each row from the previous one: pair up neighbouring entries of the previous row and wrap the sums in a 1 at each end.",
    "hint_zh": "由上一行推出下一行：把上一行相邻的两个数两两相加，再在两端各补一个 1。",
    "cases": [
        {"args": [4], "expected": [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1]]},
        {"args": [1], "expected": [[1]]},
        {"args": [2], "expected": [[1], [1, 1]]},
        {"args": [3], "expected": [[1], [1, 1], [1, 2, 1]]},
        # Updating one row in place from left to right would give [1, 3, 4, 1] as row 4
        {"args": [5], "expected": [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]},
        {"args": [7], "expected": [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1], [1, 5, 10, 10, 5, 1], [1, 6, 15, 20, 15, 6, 1]]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        rows = [[1]]
        for _ in range(numRows - 1):
            prev = rows[-1]
            # Each inner entry is the sum of two neighbours in the row above; the ends are 1
            rows.append([1] + [prev[j] + prev[j + 1] for j in range(len(prev) - 1)] + [1])
        return rows''',
    "explanation": "Each row is derived from the row above it. Every inner entry is the sum of two adjacent entries in the previous row, and the two ends are always 1. The rows are built one after another, so each row reads only the completed row before it.\n\nTime O(numRows²), space O(numRows²) for the output.",
    "explanation_zh": "每一行都由上一行推出。行内每个非首尾的数，等于上一行相邻两数之和；首尾始终是 1。逐行生成，每一行只读取已经完整的上一行。\n\n时间 O(numRows²)，空间 O(numRows²)（即输出的大小）。",
}
