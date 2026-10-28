"""Edit Distance (LeetCode 72)."""

import random


def _distance(a: str, b: str) -> int:
    # Full table, independent of the reference's single rolling row
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = i + j
            elif a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _large_case():
    # Two independent 500-letter strings over a four-letter alphabet. Trying every sequence of
    # operations is exponential in the length; the table is 250 000 cells
    rng = random.Random(72)
    a = "".join(rng.choice("abcd") for _ in range(500))
    b = "".join(rng.choice("abcd") for _ in range(500))
    return {"args": [a, b], "expected": _distance(a, b)}


def _random_case(rng):
    # word2 is word1 with some letters swapped, so a replace is the cheap edit; sometimes it
    # also gets one insert or delete. Independent random words differ in length and rarely
    # need a replace. Measured: about 8 of 20 cases expose the missing replace
    w1 = "".join(rng.choice("abc") for _ in range(rng.randint(1, 6)))
    w2 = "".join(rng.choice("abc") if rng.random() < 0.4 else ch for ch in w1)
    r = rng.random()
    if r < 0.25 and w2:
        i = rng.randrange(len(w2))
        w2 = w2[:i] + w2[i + 1:]
    elif r < 0.5:
        i = rng.randrange(len(w2) + 1)
        w2 = w2[:i] + rng.choice("abc") + w2[i:]
    return {"args": [w1, w2]}


TASK = {
    "title": "Edit Distance",
    "title_zh": "编辑距离",
    "difficulty": "Medium",
    "leetcode_id": 72,
    "leetcode_slug": "edit-distance",
    "category": "Multi-dimensional DP",
    "function_name": "minDistance",
    "params": [("word1", "str"), ("word2", "str")],
    "returns": "int",
    "description_en": "Given two strings `word1` and `word2`, return the fewest single-character operations that turn `word1` into `word2`. The allowed operations are: insert one character, delete one character, or replace one character with a different one.\n\n**Example 1:** `word1 = \"sunday\", word2 = \"saturday\"` → `3`\n\n**Example 2:** `word1 = \"flaw\", word2 = \"lawn\"` → `2` (delete `f`, then insert `n`)\n\n**Constraints:**\n- `0 <= word1.length, word2.length <= 500`\n- Both strings consist of lowercase English letters",
    "description_zh": "给定两个字符串 `word1` 和 `word2`，返回把 `word1` 变成 `word2` 所需的最少单字符操作次数。允许的操作有：插入一个字符、删除一个字符、把一个字符替换成另一个字符。\n\n**示例 1：** `word1 = \"sunday\", word2 = \"saturday\"` → `3`\n\n**示例 2：** `word1 = \"flaw\", word2 = \"lawn\"` → `2`（删除 `f`，再插入 `n`）\n\n**约束：**\n- `0 <= word1.length, word2.length <= 500`\n- 两个字符串都只由小写英文字母组成",
    "hint": "Let dp[i][j] be the fewest operations to turn the first i characters of word1 into the first j of word2. Compare the last characters: if they match, nothing is spent; otherwise take the cheapest of insert, delete and replace, plus one.",
    "hint_zh": "设 dp[i][j] 为把 word1 前 i 个字符变成 word2 前 j 个字符的最少操作数。比较末尾字符：相同则不花费；否则在插入、删除、替换三者中取最小值再加一。",
    "cases": [
        {"args": ["sunday", "saturday"], "expected": 3},
        {"args": ["flaw", "lawn"], "expected": 2},
        {"args": ["", ""], "expected": 0},
        {"args": ["", "abc"], "expected": 3},
        {"args": ["abc", ""], "expected": 3},
        # A single replacement; a solver without "replace" needs two operations here
        {"args": ["a", "b"], "expected": 1},
        {"args": ["cat", "cut"], "expected": 1},
        {"args": ["abcdef", "abcdef"], "expected": 0},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # prev[j]: distance between the processed prefix of word1 and word2[:j]
        prev = list(range(len(word2) + 1))
        for i, a in enumerate(word1, 1):
            cur = [i]
            for j, b in enumerate(word2, 1):
                if a == b:
                    cur.append(prev[j - 1])
                else:
                    # replace: prev[j - 1]; delete from word1: prev[j]; insert: cur[j - 1]
                    cur.append(1 + min(prev[j - 1], prev[j], cur[j - 1]))
            prev = cur
        return prev[-1]''',
    "explanation": "Let `dp[i][j]` be the fewest operations that turn `word1[:i]` into `word2[:j]`. With no characters left, the cost is the length of the other one. If the last characters match, they cost nothing: `dp[i][j] = dp[i-1][j-1]`. Otherwise the last character of `word1` is replaced (`dp[i-1][j-1]`), deleted (`dp[i-1][j]`), or the character of `word2` is inserted after `word1[:i]` (`dp[i][j-1]`), and the cheapest of these plus one is taken. The table is kept as one row at a time.\n\nTime O(m · n), space O(n).",
    "explanation_zh": "令 `dp[i][j]` 为把 `word1[:i]` 变成 `word2[:j]` 的最少操作数。若某一边已经没有字符，代价就是另一边的长度。若末尾字符相同，不需要操作：`dp[i][j] = dp[i-1][j-1]`。否则 `word1` 的末尾字符可以被替换（`dp[i-1][j-1]`）或删除（`dp[i-1][j]`），或者在 `word1[:i]` 后插入 `word2` 的末尾字符（`dp[i][j-1]`），取三者最小再加一。表格每次只保留一行。\n\n时间 O(m · n)，空间 O(n)。",
}
