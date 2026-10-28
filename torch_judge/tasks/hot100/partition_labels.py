"""Partition Labels (LeetCode 763)."""

import random


def _piece_sizes(s: str) -> list[int]:
    # A cut after position k is allowed when no letter of s[:k] appears again later. Cut at every
    # allowed position: the result is the finest split, which is the unique answer.
    cuts = [k for k in range(1, len(s) + 1) if all(s.rfind(c) < k for c in set(s[:k]))]
    return [b - a for a, b in zip([0] + cuts, cuts)]


def _large_case():
    # 500 characters (the maximum). Blocks of a few letters chosen at random, so the string
    # splits into several pieces at places that are hard to see by eye. Even an O(n²) search
    # (re-scanning the string for each cut) finishes in milliseconds here, so this case checks
    # correctness rather than speed.
    rng = random.Random(763)
    letters = "abcdefghijklmnopqrstuvwxyz"
    s = ""
    while len(s) < 500:
        pool = rng.sample(letters, 2)
        s += "".join(rng.choice(pool) for _ in range(rng.randint(1, 20)))
    s = s[:500]
    return {"args": [s], "expected": _piece_sizes(s)}


def _random_case(rng):
    return {"args": ["".join(rng.choice("abc") for _ in range(rng.randint(1, 12)))]}


TASK = {
    "title": "Partition Labels",
    "title_zh": "划分字母区间",
    "difficulty": "Medium",
    "leetcode_id": 763,
    "leetcode_slug": "partition-labels",
    "category": "Greedy",
    "function_name": "partitionLabels",
    "params": [("s", "str")],
    "returns": "int[]",
    "description_en": "Split the lowercase string `s` into contiguous pieces so that each letter appears in at most one piece. Make as many pieces as possible. Return the sizes of the pieces in order from left to right.\n\n**Example 1:** `s = \"abcabdde\"` → `[5,2,1]` (`abcab`, `dd`, `e`)\n\n**Example 2:** `s = \"zzz\"` → `[3]`\n\n**Constraints:**\n- `1 <= s.length <= 500`\n- `s` consists of lowercase English letters",
    "description_zh": "把小写字符串 `s` 切分成若干连续的片段，使得每个字母最多只出现在一个片段中。要求片段尽可能多。按从左到右的顺序返回每个片段的长度。\n\n**示例 1：** `s = \"abcabdde\"` → `[5,2,1]`（`abcab`、`dd`、`e`）\n\n**示例 2：** `s = \"zzz\"` → `[3]`\n\n**约束：**\n- `1 <= s.length <= 500`\n- `s` 由小写英文字母组成",
    "hint": "Record where each letter last appears. Walk left to right and keep the furthest last-position you have seen; the piece ends when your index catches up with it.",
    "hint_zh": "先记下每个字母最后一次出现的位置。从左到右遍历，记录已见字母中最远的最后位置；当下标追上这个位置时，当前片段结束。",
    "cases": [
        {"args": ["abcabdde"], "expected": [5, 2, 1]},
        {"args": ["zzz"], "expected": [3]},
        {"args": ["a"], "expected": [1]},
        # Cutting at each letter's last occurrence gives [3, 1]: the piece has to reach the
        # last b at index 3, so the whole string is one piece
        {"args": ["abab"], "expected": [4]},
        {"args": ["abcde"], "expected": [1, 1, 1, 1, 1]},
        {"args": ["aabbcc"], "expected": [2, 2, 2]},
        {"args": ["eccbbbbdec"], "expected": [10]},
        {"args": ["qiqxyzyxqz"], "expected": [10]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        last = {c: i for i, c in enumerate(s)}  # last index of each letter
        sizes, start, end = [], 0, 0
        for i, c in enumerate(s):
            end = max(end, last[c])  # this piece must reach every letter's last index
            if i == end:
                sizes.append(end - start + 1)
                start = i + 1
        return sizes''',
    "explanation": "Record each letter's last index. Scan left to right, and let the current piece reach the largest last index of any letter seen in it. When the scan reaches that point, no letter in the piece occurs later, so the piece can end there. Cutting as soon as that happens gives the most pieces.\n\nTime O(n), space O(1) for the 26 letters.",
    "explanation_zh": "先记录每个字母最后一次出现的位置。从左到右扫描，当前片段的终点取片段内所有字母最后位置的最大值。扫描到这个终点时，片段内的字母都不会再出现，于是可以在此切分。每次都尽早切分，得到的片段数最多。\n\n时间 O(n)，空间 O(1)（只有 26 个字母）。",
}
