"""Longest Palindromic Substring (LeetCode 5)."""

import random


def _longest(s: str) -> str:
    best = s[:1]
    for center in range(2 * len(s) - 1):
        lo, hi = center // 2, (center + 1) // 2
        while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
            lo, hi = lo - 1, hi + 1
        if hi - lo - 1 > len(best):
            best = s[lo + 1:hi]
    return best


def _large_case():
    # 1000 characters from a two-letter alphabet: lots of palindromes, so the O(n³) "check
    # every substring" approach does ~10^8 character comparisons
    rng = random.Random(5)
    s = "".join(rng.choice("ab") for _ in range(1000))
    return {"args": [s], "expected": _longest(s)}


def _random_case(rng):
    return {"args": ["".join(rng.choice("abc") for _ in range(rng.randint(1, 12)))]}


TASK = {
    "title": "Longest Palindromic Substring",
    "title_zh": "最长回文子串",
    "difficulty": "Medium",
    "leetcode_id": 5,
    "leetcode_slug": "longest-palindromic-substring",
    "category": "Multi-dimensional DP",
    "function_name": "longestPalindrome",
    "params": [("s", "str")],
    "returns": "str",
    # Ties are common ("ab" -> "a" or "b"), so any longest palindrome is accepted
    "check": "custom",
    "checker": '''def check(args, actual, expected):
    s = args[0]
    return (isinstance(actual, str) and len(actual) == len(expected)
            and actual == actual[::-1] and actual in s)''',
    "description_en": "Given a string `s`, return its longest substring that reads the same forwards and backwards. If several substrings tie for the longest, return any of them.\n\n**Example 1:** `s = \"racecars\"` → `\"racecar\"`\n\n**Example 2:** `s = \"xyyz\"` → `\"yy\"`\n\n**Constraints:**\n- `1 <= s.length <= 1000`\n- `s` consists of English letters and digits",
    "description_zh": "给定字符串 `s`，返回它最长的回文子串（正着读和倒着读都一样）。如果有多个子串并列最长，返回其中任意一个。\n\n**示例 1：** `s = \"racecars\"` → `\"racecar\"`\n\n**示例 2：** `s = \"xyyz\"` → `\"yy\"`\n\n**约束：**\n- `1 <= s.length <= 1000`\n- `s` 只由英文字母和数字组成",
    "hint": "Every palindrome mirrors around a center: a character (odd length) or the gap between two characters (even length). Expand outwards from each of the 2n - 1 centers.",
    "hint_zh": "每个回文串都关于一个中心对称：中心是一个字符（奇数长度）或两个字符之间的空隙（偶数长度）。从 2n - 1 个中心分别向两边扩展。",
    "cases": [
        {"args": ["racecars"], "expected": "racecar"},
        {"args": ["xyyz"], "expected": "yy"},
        {"args": ["q"], "expected": "q"},
        # Ties: any single character is a longest palindrome
        {"args": ["abcd"], "expected": "a"},
        # Even-length palindrome; expanding only around characters finds just "b"
        {"args": ["abba"], "expected": "abba"},
        {"args": ["aaaa"], "expected": "aaaa"},
        {"args": ["babad"], "expected": "bab"},
        # The longest palindrome sits at the very end
        {"args": ["xyzlevel"], "expected": "level"},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def longestPalindrome(self, s: str) -> str:
        start, length = 0, 1
        # 2n - 1 centers: index c // 2 for odd lengths, the gap after it for even ones
        for center in range(2 * len(s) - 1):
            lo, hi = center // 2, (center + 1) // 2
            while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
                lo, hi = lo - 1, hi + 1
            if hi - lo - 1 > length:
                start, length = lo + 1, hi - lo - 1
        return s[start:start + length]''',
    "explanation": "Expand around centers. A palindrome is determined by its center and radius, and there are `2n - 1` centers: each character (odd lengths) and each gap between neighbours (even lengths). From each center, grow outwards while the two ends match; the stretch reached is the longest palindrome with that center. Keep the best over all centers.\n\nThe interval DP `dp[i][j] = s[i] == s[j] and dp[i+1][j-1]` is the same O(n²) idea with O(n²) memory; Manacher's algorithm gets O(n).\n\nTime O(n²), space O(1).",
    "explanation_zh": "中心扩展法。回文串由中心和半径确定，一共有 `2n - 1` 个中心：每个字符（奇数长度）和每两个相邻字符之间的空隙（偶数长度）。从每个中心向两边扩展，只要两端字符相同就继续，得到的就是以该中心为中心的最长回文串。所有中心中取最长的。\n\n区间 DP `dp[i][j] = s[i] == s[j] and dp[i+1][j-1]` 是同样的 O(n²) 思路，但要 O(n²) 空间；Manacher 算法可以做到 O(n)。\n\n时间 O(n²)，空间 O(1)。",
}
