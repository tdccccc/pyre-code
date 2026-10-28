"""Longest Substring Without Repeating Characters (LeetCode 3)."""

import random
import string


def _longest(s: str) -> int:
    last, start, best = {}, 0, 0
    for i, ch in enumerate(s):
        if last.get(ch, -1) >= start:
            start = last[ch] + 1
        last[ch] = i
        best = max(best, i - start + 1)
    return best


def _large_case():
    rng = random.Random(3)
    alphabet = string.ascii_letters + string.digits + "!#$%&()*+-./:;<=>?@[]^_{|}~"
    s = "".join(rng.choice(alphabet) for _ in range(50000))
    return {"args": [s], "expected": _longest(s)}


TASK = {
    "title": "Longest Substring Without Repeating Characters",
    "title_zh": "无重复字符的最长子串",
    "difficulty": "Medium",
    "leetcode_id": 3,
    "leetcode_slug": "longest-substring-without-repeating-characters",
    "category": "Sliding Window",
    "function_name": "lengthOfLongestSubstring",
    "params": [("s", "str")],
    "returns": "int",
    "description_en": "Given a string `s`, return the length of its longest **substring** (a contiguous run of characters) in which no character appears twice.\n\n**Example 1:** `s = \"abacdcf\"` → `4` (the substring `\"bacd\"`)\n\n**Example 2:** `s = \"zzzz\"` → `1`\n\n**Constraints:**\n- `0 <= s.length <= 5 * 10^4`\n- `s` consists of English letters, digits, symbols and spaces",
    "description_zh": "给定字符串 `s`，返回其中不含重复字符的最长**子串**（连续的一段字符）的长度。\n\n**示例 1：** `s = \"abacdcf\"` → `4`（子串 `\"bacd\"`）\n\n**示例 2：** `s = \"zzzz\"` → `1`\n\n**约束：**\n- `0 <= s.length <= 5 * 10^4`\n- `s` 由英文字母、数字、符号和空格组成",
    "hint": "Slide a window [start, i]. Remember where each character was last seen; when s[i] repeats inside the window, jump start past that earlier position.",
    "hint_zh": "维护滑动窗口 [start, i]，记录每个字符最后出现的位置。当 s[i] 在窗口内重复时，把 start 跳到上次出现位置的后一位。",
    "cases": [
        {"args": ["abacdcf"], "expected": 4},
        {"args": ["zzzz"], "expected": 1},
        {"args": [""], "expected": 0},
        # A line holding a single space is easy to lose with input().strip(); core mode only
        {"args": [" "], "expected": 1, "acm": False},
        {"args": ["abcdeafg"], "expected": 7},
        {"args": ["abba"], "expected": 2},
    ],
    "large_case": _large_case,
    "solution": '''class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        last = {}  # char -> index where it was last seen
        start = best = 0
        for i, ch in enumerate(s):
            if last.get(ch, -1) >= start:
                start = last[ch] + 1
            last[ch] = i
            best = max(best, i - start + 1)
        return best''',
    "explanation": "Sliding window. `start` is the left edge of a window with no repeats. When `s[i]` was last seen at or after `start`, the window must begin right after that occurrence. The check `>= start` matters: an occurrence left of the window (like the first `a` in `\"abba\"`) must not move `start` backwards.\n\nTime O(n), space O(size of the character set).",
    "explanation_zh": "滑动窗口。`start` 是无重复窗口的左边界。如果 `s[i]` 上次出现的位置在 `start` 及之后，窗口就得从那次出现的下一位开始。判断 `>= start` 很关键：窗口左边的旧位置（比如 `\"abba\"` 里第一个 `a`）不能让 `start` 往回退。\n\n时间 O(n)，空间 O(字符集大小)。",
}
