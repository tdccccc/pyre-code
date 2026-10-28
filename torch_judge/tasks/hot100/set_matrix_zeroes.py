"""Set Matrix Zeroes (LeetCode 73)."""

import random


def _zeroed(matrix):
    # Collect the rows and columns that hold a zero, then clear them in a fresh copy
    rows = {i for i, row in enumerate(matrix) if 0 in row}
    cols = {j for row in matrix for j, v in enumerate(row) if v == 0}
    return [[0 if i in rows or j in cols else v for j, v in enumerate(row)]
            for i, row in enumerate(matrix)]


def _large_case():
    # 200 x 200 with about 200 zeros scattered across most rows and columns
    rng = random.Random(73)
    matrix = [[0 if rng.random() < 1 / 300 else rng.randint(-10**9, 10**9) for _ in range(200)]
              for _ in range(200)]
    return {"args": [matrix], "expected": _zeroed(matrix)}


def _random_case(rng):
    m, n = rng.randint(1, 4), rng.randint(1, 4)
    return {"args": [[[rng.choice([0, 1, 2, -3, 5]) for _ in range(n)] for _ in range(m)]]}


TASK = {
    "title": "Set Matrix Zeroes",
    "title_zh": "矩阵置零",
    "difficulty": "Medium",
    "leetcode_id": 73,
    "leetcode_slug": "set-matrix-zeroes",
    "category": "Matrix",
    "function_name": "setZeroes",
    "params": [("matrix", "int[][]")],
    "returns": "void",
    "inplace": "matrix",
    "description_en": "Given an `m x n` integer matrix, find every `0` and set its entire row and its entire column to `0`. Modify the matrix **in place**; do not return anything.\n\n**Example 1:** `matrix = [[2,0,3],[4,5,6]]` → `[[0,0,0],[4,0,6]]`\n\n**Example 2:** `matrix = [[1,2],[3,0],[4,5],[6,7]]` → `[[1,0],[0,0],[4,0],[6,0]]`\n\n**Constraints:**\n- `1 <= m, n <= 200`\n- `-10^9 <= matrix[i][j] <= 10^9`\n\n**Follow-up:** can you use only O(1) extra space?",
    "description_zh": "给定 `m x n` 的整数矩阵，找出所有值为 `0` 的元素，并把它所在的整行和整列全部置为 `0`。必须**原地**修改矩阵，不要返回任何值。\n\n**示例 1：** `matrix = [[2,0,3],[4,5,6]]` → `[[0,0,0],[4,0,6]]`\n\n**示例 2：** `matrix = [[1,2],[3,0],[4,5],[6,7]]` → `[[1,0],[0,0],[4,0],[6,0]]`\n\n**约束：**\n- `1 <= m, n <= 200`\n- `-10^9 <= matrix[i][j] <= 10^9`\n\n**进阶：** 你能只使用 O(1) 的额外空间完成吗？",
    "hint": "Don't clear rows and columns while you scan, or the zeros you create will spread. Note which rows and columns hold a zero first, then clear them at the end.",
    "hint_zh": "不要边扫描边清零，否则新产生的零会继续扩散。先记下哪些行和哪些列含有零，最后再统一清零。",
    "cases": [
        {"args": [[[2, 0, 3], [4, 5, 6]]], "expected": [[0, 0, 0], [4, 0, 6]]},
        {"args": [[[1, 2], [3, 0], [4, 5], [6, 7]]], "expected": [[1, 0], [0, 0], [4, 0], [6, 0]]},
        {"args": [[[1, 2, 3], [4, 5, 6], [7, 8, 0]]], "expected": [[1, 2, 0], [4, 5, 0], [0, 0, 0]]},
        # The zero at (0, 0) clears column 0 and row 0; the zero it causes at (1, 0) must not
        # trigger more clearing. Spreading zeros as you find them turns this into all zeros
        {"args": [[[0, 1], [1, 1]]], "expected": [[0, 0], [0, 1]]},
        {"args": [[[1, 0, 3, 4]]], "expected": [[0, 0, 0, 0]]},
        {"args": [[[7], [0], [9]]], "expected": [[0], [0], [0]]},
        {"args": [[[5]]], "expected": [[5]]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        m, n = len(matrix), len(matrix[0])
        first_row_zero = any(v == 0 for v in matrix[0])
        first_col_zero = any(matrix[i][0] == 0 for i in range(m))
        # Row 0 and column 0 act as markers for the interior rows and columns
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        if first_row_zero:
            for j in range(n):
                matrix[0][j] = 0
        if first_col_zero:
            for i in range(m):
                matrix[i][0] = 0''',
    "explanation": "Use the first row and first column as markers: `matrix[i][0] = 0` means row `i` must be cleared, and `matrix[0][j] = 0` means column `j` must be cleared. Those two border lines are also zeroed by the markers, so first record whether they held a zero at the start. Fill the interior from the markers, then clear the first row and column if their flags say so.\n\nTime O(mn), space O(1).",
    "explanation_zh": "用第一行和第一列当作标记：`matrix[i][0] = 0` 表示第 `i` 行要清零，`matrix[0][j] = 0` 表示第 `j` 列要清零。但这两条边本身也会被标记覆盖，所以要先记下它们原来是否含零。然后根据标记填充内部，最后再根据标志清空第一行和第一列。\n\n时间 O(mn)，空间 O(1)。",
}
