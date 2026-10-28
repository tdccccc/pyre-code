"""Longest Valid Parentheses (LeetCode 32)."""

import random


def _longest(s: str) -> int:
    # dp[i]: length of the longest valid substring that ends at index i
    dp = [0] * len(s)
    best = 0
    for i in range(1, len(s)):
        if s[i] == ")":
            j = i - dp[i - 1] - 1  # the '(' that would pair with this ')'
            if j >= 0 and s[j] == "(":
                dp[i] = dp[i - 1] + 2 + (dp[j - 1] if j > 0 else 0)
            best = max(best, dp[i])
    return best


def _large_case():
    # 30000 random brackets. Checking every substring for validity takes ~4.5·10⁸ steps,
    # far past the time limit; the stack reference is linear
    rng = random.Random(32)
    s = "".join(rng.choice("()") for _ in range(30000))
    return {"args": [s], "expected": _longest(s)}


def _random_case(rng):
    return {"args": ["".join(rng.choice("()") for _ in range(rng.randint(0, 12)))]}


TASK = {
    "title": "Longest Valid Parentheses",
    "title_zh": "最长有效括号",
    "difficulty": "Hard",
    "leetcode_id": 32,
    "leetcode_slug": "longest-valid-parentheses",
    "category": "Dynamic Programming",
    "function_name": "longestValidParentheses",
    "params": [("s", "str")],
    "returns": "int",
    "description_en": "Given a string `s` made only of `(` and `)`, return the length of the longest substring that is a valid bracket sequence: every `(` is closed by a later `)`, and the pairs nest properly without crossing. The empty string gives `0`.\n\n**Example 1:** `s = \"(()))(\"` → `4` (the substring `(())`)\n\n**Example 2:** `s = \"))()()\"` → `4` (the substring `()()`)\n\n**Constraints:**\n- `0 <= s.length <= 3 * 10^4`\n- `s[i]` is `(` or `)`",
    "description_zh": "给定只由字符 `(` 和 `)` 组成的字符串 `s`，返回最长有效括号子串的长度。有效括号子串指：每个 `(` 都能与其后某个 `)` 配对，且所有配对互相嵌套、不交叉。空字符串的结果是 `0`。\n\n**示例 1：** `s = \"(()))(\"` → `4`（子串 `(())`）\n\n**示例 2：** `s = \"))()()\"` → `4`（子串 `()()`）\n\n**约束：**\n- `0 <= s.length <= 3 * 10^4`\n- `s[i]` 为 `(` 或 `)`",
    "hint": "Keep a stack of indices. Push each '('. On ')', pop; if the stack is now empty, this ')' is unmatched and becomes the new starting point. Otherwise the valid run ends here and starts after the index on top of the stack.",
    "hint_zh": "用栈保存下标。遇到 '(' 就入栈；遇到 ')' 就出栈。如果出栈后栈空，说明这个 ')' 没有配对，它成为新的起点。否则有效串在此结束，起点是栈顶下标的下一位。",
    "cases": [
        {"args": ["(()))("], "expected": 4},
        {"args": ["))()()"], "expected": 4},
        {"args": [""], "expected": 0},
        {"args": ["("], "expected": 0},
        {"args": [")("], "expected": 0},
        {"args": ["()"], "expected": 2},
        # Two adjacent valid pairs merge into one run of length 4
        {"args": ["((()()"], "expected": 4},
        # The unmatched ')' at indices 0 and 1 must move the starting point; keeping the old
        # base (-1) would count them and report 8
        {"args": ["))()()()"], "expected": 6},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def longestValidParentheses(self, s: str) -> int:
        best = 0
        # Indices of unmatched characters, plus the index just before the current run
        stack = [-1]
        for i, ch in enumerate(s):
            if ch == "(":
                stack.append(i)
            else:
                stack.pop()
                if stack:
                    best = max(best, i - stack[-1])
                else:
                    stack.append(i)  # unmatched ')': it is the new base for later runs
        return best''',
    "explanation": "Track the indices of unmatched characters on a stack, with `-1` at the bottom as a base. A `(` pushes its index. A `)` pops the last index: if the stack is still non-empty, the valid run ending here starts right after the new top, so its length is `i - stack[-1]`. If the stack became empty, the `)` had nothing to match and its index becomes the new base.\n\nTime O(n), space O(n).",
    "explanation_zh": "用栈记录未匹配字符的下标，栈底放一个 `-1` 作为基准。遇到 `(` 就把它的下标入栈。遇到 `)` 就弹出栈顶：如果弹出后栈非空，以当前位置结尾的有效串从新栈顶之后开始，长度为 `i - stack[-1]`。如果栈空了，说明这个 `)` 没有可配对的字符，它的下标成为新的基准。\n\n时间 O(n)，空间 O(n)。",
}
