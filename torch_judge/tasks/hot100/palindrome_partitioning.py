"""Palindrome Partitioning (LeetCode 131)."""

import random


def _partitions(s: str) -> list:
    # Every split into palindromes, by plain recursion; written only to produce the expected answer
    out, pieces = [], []

    def go(start: int) -> None:
        if start == len(s):
            out.append(list(pieces))
            return
        for end in range(start + 1, len(s) + 1):
            piece = s[start:end]
            if piece == piece[::-1]:
                pieces.append(piece)
                go(end)
                pieces.pop()

    go(0)
    return out


def _large_case():
    # At the constraint limit (14 characters) with mostly "a", so most splits are palindromes:
    # several thousand answers. The output size is the cost, so the expected answers are
    # built by plain recursion, not from the reference's palindrome table.
    rng = random.Random(131)
    s = "".join(rng.choice("aaaaaaaab") for _ in range(14))
    return {"args": [s], "expected": _partitions(s)}


def _random_case(rng):
    return {"args": ["".join(rng.choice("ab") for _ in range(rng.randint(1, 6)))]}


TASK = {
    "title": "Palindrome Partitioning",
    "title_zh": "分割回文串",
    "difficulty": "Medium",
    "leetcode_id": 131,
    "leetcode_slug": "palindrome-partitioning",
    "category": "Backtracking",
    "function_name": "partition",
    "params": [("s", "str")],
    "returns": "str[][]",
    "check": "unordered",
    "description_en": """Given a string `s`, split it into one or more consecutive pieces so that every piece is a palindrome (it reads the same forwards and backwards). Return every possible way to split `s`. Each way is a list of its pieces in left-to-right order. The ways may be returned in any order.

**Example 1:** `s = "aba"` → `[["a","b","a"],["aba"]]`

**Example 2:** `s = "abc"` → `[["a","b","c"]]`

**Constraints:**
- $1 \\le \\texttt{s.length} \\le 14$
- `s` consists of lowercase English letters only""",
    "description_zh": """给定字符串 `s`，把它切分成若干段连续的子串，使每一段都是回文串（正着读和倒着读都一样）。返回 `s` 所有可能的切分方案。每种方案是按从左到右顺序排列的各段子串组成的列表。方案的顺序任意。

**示例 1：** `s = "aba"` → `[["a","b","a"],["aba"]]`

**示例 2：** `s = "abc"` → `[["a","b","c"]]`

**约束：**
- $1 \\le \\texttt{s.length} \\le 14$
- `s` 只包含小写英文字母""",
    "hint": "Choose the first piece as a palindrome starting at position 0, then solve the rest of the string the same way. Precompute which substrings are palindromes so each check is a lookup.",
    "hint_zh": "从位置 0 开始选一段回文串作为第一段，然后用同样的方法处理剩下的部分。可以预先算好每个子串是否为回文，使每次判断只需查表。",
    "cases": [
        {"args": ["aba"], "expected": [["a", "b", "a"], ["aba"]]},
        {"args": ["abc"], "expected": [["a", "b", "c"]]},
        {"args": ["a"], "expected": [["a"]]},
        # The whole string is one piece when it is a palindrome; an off-by-one that stops
        # short of the end misses ["aa"] here
        {"args": ["aa"], "expected": [["a", "a"], ["aa"]]},
        {"args": ["aab"], "expected": [["a", "a", "b"], ["aa", "b"]]},
        {"args": ["abba"], "expected": [["a", "b", "b", "a"], ["a", "bb", "a"], ["abba"]]},
        # "xyzzy" itself is not a palindrome, so it cannot be one piece; "yzzy" can
        {"args": ["xyzzy"], "expected": [["x", "y", "z", "z", "y"], ["x", "y", "zz", "y"], ["x", "yzzy"]]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)
        # is_pal[i][j]: s[i..j] reads the same forwards and backwards
        is_pal = [[False] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                is_pal[i][j] = s[i] == s[j] and (j - i < 2 or is_pal[i + 1][j - 1])

        result, pieces = [], []

        def backtrack(start: int) -> None:
            if start == n:
                result.append(pieces[:])
                return
            for end in range(start, n):
                if is_pal[start][end]:
                    pieces.append(s[start:end + 1])
                    backtrack(end + 1)
                    pieces.pop()

        backtrack(0)
        return result''',
    "explanation": "Backtracking over the cut points. From the current start position, try every end position where `s[start..end]` is a palindrome, take that piece, and recurse from the position after it. When the start reaches the end of the string, the pieces taken so far form one valid split. Palindrome checks are precomputed in a table `is_pal[i][j]`, filled from the inside out: `s[i..j]` is a palindrome when its ends match and the part inside it is one too.\n\nTime O(n² + n · 2^n) (the table, then up to 2^(n-1) splits of length up to n), space O(n²) for the table.",
    "explanation_zh": "对切分点做回溯。从当前起点出发，尝试每一个使 `s[start..end]` 为回文的终点，取出该段，然后从其后一个位置继续递归。起点走到字符串末尾时，已取出的各段就是一种合法切分。回文判断预先存在表 `is_pal[i][j]` 中，由内向外填表：两端字符相同且中间部分也是回文，则整段是回文。\n\n时间 O(n² + n · 2^n)（建表，再加上至多 2^(n-1) 种切分），空间 O(n²)（回文表）。",
}
