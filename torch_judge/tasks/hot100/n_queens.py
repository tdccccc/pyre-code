"""N-Queens (LeetCode 51)."""

import random


def _boards(n: int) -> list:
    # Row by row, trying each column; a queen is safe when no earlier queen shares its column
    # or a diagonal. Written only to produce the expected answer.
    out, cols = [], []

    def place(row: int) -> None:
        if row == n:
            out.append(["." * c + "Q" + "." * (n - c - 1) for c in cols])
            return
        for c in range(n):
            if all(c != cc and abs(c - cc) != row - r for r, cc in enumerate(cols)):
                cols.append(c)
                place(row + 1)
                cols.pop()

    place(0)
    return out


def _large_case():
    # At the constraint limit: n = 9 has 352 placements. Backtracking with constant-time
    # checks visits a few thousand partial boards.
    return {"args": [9], "expected": _boards(9)}


def _random_case(rng):
    return {"args": [rng.randint(1, 5)]}


TASK = {
    "title": "N-Queens",
    "title_zh": "N 皇后",
    "difficulty": "Hard",
    "leetcode_id": 51,
    "leetcode_slug": "n-queens",
    "category": "Backtracking",
    "function_name": "solveNQueens",
    "params": [("n", "int")],
    "returns": "str[][]",
    "check": "unordered",
    "description_en": """Place `n` queens on an `n × n` chessboard so that no two queens attack each other. Queens attack along their row, their column, and both diagonals, so no two queens may share a row, a column, or a diagonal.

Return every valid placement. A placement is a list of `n` strings, one per row, where `Q` is a queen and `.` is an empty square. The placements may be returned in any order.

**Example 1:** `n = 1` → `[["Q"]]`

**Example 2:** `n = 2` → `[]`

**Constraints:**
- $1 \\le \\texttt{n} \\le 9$""",
    "description_zh": """在 `n × n` 的棋盘上放置 `n` 个皇后，使任意两个皇后都不能互相攻击。皇后可以沿所在的行、列和两条斜线攻击，因此任意两个皇后都不能同行、同列或同一条斜线。

返回所有合法的摆法。每种摆法是一个由 `n` 个字符串组成的列表，每个字符串对应一行：`Q` 表示皇后，`.` 表示空格。摆法的顺序任意。

**示例 1：** `n = 1` → `[["Q"]]`

**示例 2：** `n = 2` → `[]`

**约束：**
- $1 \\le \\texttt{n} \\le 9$""",
    "hint": "Place one queen per row. Keep three sets: the columns in use, the \"/\" diagonals (row + col) and the \"\\\" diagonals (row - col). A column is safe when it is in none of the sets.",
    "hint_zh": "每行放一个皇后。维护三个集合：已占用的列、「/」斜线（行 + 列）和「\\」斜线（行 - 列）。某一列三个集合都不包含它时才能放。",
    "cases": [
        {"args": [1], "expected": [["Q"]]},
        {"args": [2], "expected": []},
        # Three queens cannot fit: a solution that checks only columns and one diagonal reports boards here
        {"args": [3], "expected": []},
        {"args": [4], "expected": [[".Q..", "...Q", "Q...", "..Q."], ["..Q.", "Q...", "...Q", ".Q.."]]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        result, cols = [], []  # cols[r] is the column of the queen placed in row r
        used_cols, used_diag, used_anti = set(), set(), set()

        def place(row: int) -> None:
            if row == n:
                result.append(["." * c + "Q" + "." * (n - c - 1) for c in cols])
                return
            for c in range(n):
                # Same column, same diagonal (row - c) or same anti-diagonal (row + c)?
                if c in used_cols or row - c in used_diag or row + c in used_anti:
                    continue
                cols.append(c)
                used_cols.add(c)
                used_diag.add(row - c)
                used_anti.add(row + c)
                place(row + 1)
                cols.pop()
                used_cols.remove(c)
                used_diag.remove(row - c)
                used_anti.remove(row + c)

        place(0)
        return result''',
    "explanation": "Place queens one row at a time. A column is free for the current row when no earlier queen uses that column, the same diagonal (row − col is constant along it) or the same anti-diagonal (row + col is constant). Three sets track those, so each check is O(1). Each full placement is converted to the list of strings when the last row is filled.\n\nTime O(n!) in the worst case with the pruning giving far fewer partial boards in practice, space O(n) for the sets and the recursion.",
    "explanation_zh": "逐行放置皇后。某列在当前行可用，当且仅当此前没有皇后占用该列，也没有皇后位于同一条主对角线（行 − 列 相同）或同一条副对角线（行 + 列 相同）上。用三个集合记录这些占用情况，每次判断 O(1)。所有行放完后，把每个摆法转换为字符串列表。\n\n最坏时间 O(n!)，剪枝后实际访问的局面远少于此；空间 O(n)（集合与递归）。",
}
