"""Decode String (LeetCode 394)."""

import random


def _decode(s: str) -> str:
    # Recursive descent: a repeat "k[...]" decodes its body once and repeats it k times
    pos = 0

    def parse() -> str:
        nonlocal pos
        out = []
        while pos < len(s) and s[pos] != "]":
            if s[pos].isdigit():
                start = pos
                while s[pos].isdigit():
                    pos += 1
                count = int(s[start:pos])
                pos += 1  # the '['
                out.append(parse() * count)
                pos += 1  # the ']'
            else:
                out.append(s[pos])
                pos += 1
        return "".join(out)

    return parse()


def _large_case():
    # The constraints cap the input at 30 characters, so this is the longest allowed form:
    # nested repeats, a two-digit count, and output of about 1000 characters
    s = "100[ab2[c3[d]]]9[ef]"
    return {"args": [s], "expected": _decode(s)}


def _encode(rng, depth=0):
    parts = []
    for _ in range(rng.randint(1, 3)):
        if depth < 2 and rng.random() < 0.4:
            parts.append(f"{rng.choice([2, 3, 10, 12])}[{_encode(rng, depth + 1)}]")
        else:
            parts.append("".join(rng.choice("abc") for _ in range(rng.randint(1, 2))))
    return "".join(parts)


def _random_case(rng):
    while True:
        s = _encode(rng)
        if len(s) <= 30:
            return {"args": [s]}


TASK = {
    "title": "Decode String",
    "title_zh": "字符串解码",
    "difficulty": "Medium",
    "leetcode_id": 394,
    "leetcode_slug": "decode-string",
    "category": "Stack",
    "function_name": "decodeString",
    "params": [("s", "str")],
    "returns": "str",
    "description_en": "An encoded string is written as `k[body]`, which means the `body` is repeated exactly `k` times. Bodies can contain letters and further encoded parts, so the nesting can go several levels deep.\n\nGiven an encoded string `s`, return the decoded string.\n\n**Example 1:** `s = \"2[ab]c3[x]\"` → `\"ababcxxx\"`\n\n**Example 2:** `s = \"a2[b2[c]]d\"` → `\"abccbccd\"`\n\n**Constraints:**\n- $1 \\le \\texttt{s.length} \\le 30$\n- `s` contains lowercase letters, digits and square brackets only\n- The input is always well formed, and every repeat count is between `1` and `300`",
    "description_zh": "一个编码字符串的格式为 `k[body]`，表示 `body` 恰好重复 `k` 次。`body` 中可以包含字母和其他编码部分，所以嵌套可以有好几层。\n\n给定编码字符串 `s`，返回解码后的字符串。\n\n**示例 1：** `s = \"2[ab]c3[x]\"` → `\"ababcxxx\"`\n\n**示例 2：** `s = \"a2[b2[c]]d\"` → `\"abccbccd\"`\n\n**约束：**\n- $1 \\le \\texttt{s.length} \\le 30$\n- `s` 只包含小写字母、数字和方括号\n- 输入总是格式正确的，且每个重复次数都在 `1` 到 `300` 之间",
    "hint": "Use a stack to remember, for each open bracket, the text decoded so far and its repeat count. On `]`, pop them and join the text with the repeated body. A repeat count can have several digits, so accumulate the digits.",
    "hint_zh": "用栈记录每个左括号之前已解码的文本和对应的重复次数。遇到 `]` 时弹出它们，把已解码文本与重复后的 body 拼接。重复次数可能有多位数字，所以要把数字逐位累加。",
    "cases": [
        {"args": ["2[ab]c3[x]"], "expected": "ababcxxx"},
        {"args": ["a2[b2[c]]d"], "expected": "abccbccd"},
        # A two-digit count: reading one digit at a time keeps only the last digit, 0
        {"args": ["10[a]"], "expected": "aaaaaaaaaa"},
        {"args": ["abc"], "expected": "abc"},
        {"args": ["2[3[a]b]"], "expected": "aaabaaab"},
        {"args": ["12[x]y"], "expected": "xxxxxxxxxxxxy"},
        {"args": ["3[a2[c]]"], "expected": "accaccacc"},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def decodeString(self, s: str) -> str:
        stack = []  # (text decoded before the bracket, repeat count) for each open bracket
        cur, num = "", 0
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)
            elif ch == "[":
                stack.append((cur, num))
                cur, num = "", 0
            elif ch == "]":
                prev, times = stack.pop()
                cur = prev + cur * times
            else:
                cur += ch
        return cur''',
    "explanation": "Stack-based decoding. Letters are appended to the current text. A digit extends the pending repeat count, so `10` is read as ten. On `[` the current text and the count are pushed and both reset. On `]` the body is repeated and appended to the text saved before its bracket.\n\nTime O(n + output length), space O(n).",
    "explanation_zh": "基于栈的解码。字母直接加入当前文本。数字会扩展待用的重复次数，所以 `10` 被读作十。遇到 `[` 时，把当前文本和重复次数压栈，然后两者清零。遇到 `]` 时，把 body 重复后接到压栈保存的前缀文本后面。\n\n时间 O(n + 输出长度)，空间 O(n)。",
}
