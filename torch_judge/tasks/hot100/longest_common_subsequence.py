"""Longest Common Subsequence (LeetCode 1143)."""

import random


def _lcs_length(a: str, b: str) -> int:
    # Bit-parallel LCS (Allison-Dix / Hyyrö): one bit per position of a, one pass over b.
    # A different method from the reference's table, and fast enough for 1000 × 1000
    masks: dict[str, int] = {}
    for i, ch in enumerate(a):
        masks[ch] = masks.get(ch, 0) | (1 << i)
    full = (1 << len(a)) - 1
    v = full
    for ch in b:
        u = v & masks.get(ch, 0)
        v = ((v + u) | (v - u)) & full
    return len(a) - bin(v).count("1")


def _large_case():
    # Two 1000-character strings over a four-letter alphabet. Trying every subsequence of
    # either string is 2^1000 candidates, so only a polynomial method can finish
    rng = random.Random(1143)
    a = "".join(rng.choice("abcd") for _ in range(1000))
    b = "".join(rng.choice("abcd") for _ in range(1000))
    return {"args": [a, b], "expected": _lcs_length(a, b)}


def _random_case(rng):
    return {"args": [
        "".join(rng.choice("ab") for _ in range(rng.randint(1, 8))),
        "".join(rng.choice("ab") for _ in range(rng.randint(1, 8))),
    ]}


TASK = {
    "title": "Longest Common Subsequence",
    "title_zh": "最长公共子序列",
    "difficulty": "Medium",
    "leetcode_id": 1143,
    "leetcode_slug": "longest-common-subsequence",
    "category": "Multi-dimensional DP",
    "function_name": "longestCommonSubsequence",
    "params": [("text1", "str"), ("text2", "str")],
    "returns": "int",
    "description_en": "Given two strings `text1` and `text2`, return the length of their longest common subsequence, or `0` if they share no characters in order. A subsequence keeps the original order of characters but may skip any of them.\n\n**Example 1:** `text1 = \"pqrsta\", text2 = \"tqsa\"` → `3` (for example `\"qsa\"`)\n\n**Example 2:** `text1 = \"xyz\", text2 = \"zyx\"` → `1`\n\n**Constraints:**\n- $1 \\le \\texttt{text1.length},\\ \\texttt{text2.length} \\le 1000$\n- Both strings consist of lowercase English letters",
    "description_zh": "给定两个字符串 `text1` 和 `text2`，返回它们最长公共子序列的长度；若没有任何按序相同的字符，返回 `0`。子序列保持字符的原有顺序，但可以跳过任意字符。\n\n**示例 1：** `text1 = \"pqrsta\", text2 = \"tqsa\"` → `3`（例如 `\"qsa\"`）\n\n**示例 2：** `text1 = \"xyz\", text2 = \"zyx\"` → `1`\n\n**约束：**\n- $1 \\le \\texttt{text1.length},\\ \\texttt{text2.length} \\le 1000$\n- 两个字符串都只由小写英文字母组成",
    "hint": "Let dp[i][j] be the answer for the first i characters of text1 and the first j of text2. If the last characters match, extend the answer for the shorter prefixes; otherwise drop one of them.",
    "hint_zh": "设 dp[i][j] 为 text1 前 i 个字符与 text2 前 j 个字符的答案。若两者末尾字符相同，就在更短前缀的答案上加一；否则舍去其中一个字符，取较大者。",
    "cases": [
        {"args": ["pqrsta", "tqsa"], "expected": 3},
        {"args": ["xyz", "zyx"], "expected": 1},
        {"args": ["a", "a"], "expected": 1},
        {"args": ["pqr", "stu"], "expected": 0},
        {"args": ["aaaa", "aa"], "expected": 2},
        # Subsequence, not substring: "abd" is common, but the longest common substring is one letter
        {"args": ["abcd", "acbd"], "expected": 3},
        {"args": ["ab", "ba"], "expected": 1},
        # The same letters in reverse order share only one letter
        {"args": ["mnopq", "qpnm"], "expected": 1},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # prev[j]: LCS length of text1's processed prefix and text2[:j]
        prev = [0] * (len(text2) + 1)
        for a in text1:
            cur = [0]
            for j, b in enumerate(text2, 1):
                if a == b:
                    cur.append(prev[j - 1] + 1)
                else:
                    cur.append(max(prev[j], cur[j - 1]))
            prev = cur
        return prev[-1]''',
    "explanation": "Let `dp[i][j]` be the LCS length of `text1[:i]` and `text2[:j]`. If the characters `text1[i-1]` and `text2[j-1]` are equal, they can both be the last character of an optimal common subsequence, so `dp[i][j] = dp[i-1][j-1] + 1`. Otherwise at least one of them is unused, so `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`. Only the previous row is needed, so the table is kept as two lists.\n\nTime O(m · n), space O(n).",
    "explanation_zh": "令 `dp[i][j]` 为 `text1[:i]` 与 `text2[:j]` 的最长公共子序列长度。若 `text1[i-1]` 与 `text2[j-1]` 相等，它们可以同时作为最优公共子序列的末尾字符，所以 `dp[i][j] = dp[i-1][j-1] + 1`。否则两者至少有一个不被使用，于是 `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`。只需要上一行，因此用两个列表来保存表格。\n\n时间 O(m · n)，空间 O(n)。",
}
