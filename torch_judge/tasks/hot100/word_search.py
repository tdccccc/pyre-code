"""Word Search (LeetCode 79)."""

import random


def _around(r: int, c: int, rows: int, cols: int):
    for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _exists(board: list, word: str) -> bool:
    # Plain depth-first search; written only to produce the expected answer
    rows, cols = len(board), len(board[0])

    def search(r: int, c: int, k: int, seen: set) -> bool:
        if board[r][c] != word[k]:
            return False
        if k == len(word) - 1:
            return True
        seen.add((r, c))
        found = any(search(nr, nc, k + 1, seen) for nr, nc in _around(r, c, rows, cols) if (nr, nc) not in seen)
        seen.discard((r, c))
        return found

    return any(search(r, c, 0, set()) for r in range(rows) for c in range(cols))


def _walk(rng, board: list, length: int, start=None) -> str:
    # A random path of `length` distinct cells: the letters along it spell a word that exists
    rows, cols = len(board), len(board[0])
    while True:
        r, c = start if start is not None else (rng.randrange(rows), rng.randrange(cols))
        cells, word = [(r, c)], board[r][c]
        while len(cells) < length:
            options = [(nr, nc) for nr, nc in _around(r, c, rows, cols) if (nr, nc) not in cells]
            if not options:
                break
            r, c = rng.choice(options)
            cells.append((r, c))
            word += board[r][c]
        if len(cells) == length:
            return word


def _large_case():
    # At the constraint limits: a 6 x 6 board and a 15-letter word. The board is mostly "A", so
    # long prefixes match and many paths are explored. The last letter appears nowhere on the
    # board, so the search has to explore every matching prefix before it answers False.
    rng = random.Random(79)
    board = [["A" if rng.random() < 0.85 else "B" for _ in range(6)] for _ in range(6)]
    word = _walk(rng, board, 14, start=(5, 5)) + "C"
    return {"args": [board, word], "expected": _exists(board, word)}


def _random_case(rng):
    # Boards with mostly "A", and long words. 85% of the words are walks over the board, so they
    # exist. On these, the search wanders into dead ends, and a cell it marked on a failed branch
    # is then wrongly blocked for a later path that needs it
    rows, cols = rng.randint(3, 4), rng.randint(3, 4)
    board = [["A" if rng.random() < 0.6 else "B" for _ in range(cols)] for _ in range(rows)]
    if rng.random() < 0.85:
        word = _walk(rng, board, rng.randint(5, 9))
    else:
        word = "".join(rng.choice("AB") for _ in range(rng.randint(3, 6)))
    return {"args": [board, word]}


TASK = {
    "title": "Word Search",
    "title_zh": "单词搜索",
    "difficulty": "Medium",
    "leetcode_id": 79,
    "leetcode_slug": "word-search",
    "category": "Backtracking",
    "function_name": "exist",
    "params": [("board", "char[][]"), ("word", "str")],
    "returns": "bool",
    "description_en": """Given an `m × n` grid of uppercase letters `board` and a string `word`, decide whether `word` can be read off the grid. The letters of `word` are read from a path of cells in which each cell is next to the previous one (up, down, left or right), and no cell is used twice.

**Example 1:** `board = [["A","B"],["C","D"]]`, `word = "ABDC"` → `true` (A → B → D → C)

**Example 2:** the same board with `word = "AA"` → `false` (there is only one A, and a cell cannot be reused)

**Constraints:**
- $1 \\le \\texttt{m},\\ \\texttt{n} \\le 6$
- $1 \\le \\texttt{word.length} \\le 15$
- `board` and `word` consist of uppercase English letters only""",
    "description_zh": """给定一个 `m × n` 的大写字母网格 `board` 和一个字符串 `word`，判断 `word` 能否在网格中找到。依次读出 `word` 的字母时，所经过的相邻格子必须上下左右相连，并且同一个格子不能使用两次。

**示例 1：** `board = [["A","B"],["C","D"]]`，`word = "ABDC"` → `true`（A → B → D → C）

**示例 2：** 同一网格，`word = "AA"` → `false`（只有一个 A，且同一格子不能重复使用）

**约束：**
- $1 \\le \\texttt{m},\\ \\texttt{n} \\le 6$
- $1 \\le \\texttt{word.length} \\le 15$
- `board` 和 `word` 只包含大写英文字母""",
    "hint": "Try each cell as the first letter, then search the four neighbours for the next letter. Mark a cell as used while the path goes through it, and unmark it when that branch fails, so other branches can use it.",
    "hint_zh": "从每个格子出发尝试作为第一个字母，然后在四个相邻格子中找下一个字母。路径经过时标记该格子为已用；某个分支失败时取消标记，好让其他分支使用它。",
    "cases": [
        {"args": [[["A", "B"], ["C", "D"]], "ABDC"], "expected": True},
        {"args": [[["A", "B"], ["C", "D"]], "AA"], "expected": False},
        {"args": [[["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"]], "CFEH"], "expected": True},
        {"args": [[["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"]], "BEDG"], "expected": True},
        # B has no D next to it
        {"args": [[["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"]], "BD"], "expected": False},
        # The first A is a dead end; the search has to restart from the second A
        {"args": [[["A", "A", "B"], ["B", "C", "A"]], "AAB"], "expected": True},
        {"args": [[["A", "A", "B"], ["B", "C", "A"]], "ACB"], "expected": True},
        # Two B's are never next to each other on this board
        {"args": [[["A", "A", "B"], ["B", "C", "A"]], "ABB"], "expected": False},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        rows, cols = len(board), len(board[0])
        visited = set()

        def dfs(r: int, c: int, k: int) -> bool:
            # Can word[k:] be read starting at (r, c), without reusing visited cells?
            if board[r][c] != word[k]:
                return False
            if k == len(word) - 1:
                return True
            visited.add((r, c))
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    if dfs(nr, nc, k + 1):
                        return True
            visited.discard((r, c))  # undo the mark: this cell may be needed by another branch
            return False

        return any(dfs(r, c, 0) for r in range(rows) for c in range(cols))''',
    "explanation": "Depth-first search from every cell whose letter matches the first letter of the word. From a cell, step to each unvisited neighbour that matches the next letter. A cell is marked as visited while it is on the current path and unmarked when the branch fails, which is what lets later branches reuse it. Success is reported as soon as the last letter matches.\n\nTime O(m · n · 4 · 3^(L-1)) in the worst case for a word of length L (each step has at most three new neighbours), space O(L) for the recursion and the visited set.",
    "explanation_zh": "从每个首字母匹配的格子出发做深度优先搜索。从当前格子出发，走向每个未访问且字母匹配下一个字母的相邻格子。格子在当前路径上时标记为已访问；分支失败时取消标记，使其他分支可以再次使用它。最后一个字母匹配即返回成功。\n\n最坏时间 O(m · n · 4 · 3^(L-1))（L 为单词长度，每步最多 3 个新方向），空间 O(L)（递归与已访问集合）。",
}
