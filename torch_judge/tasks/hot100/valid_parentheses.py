"""Valid Parentheses (LeetCode 20)."""

import random

MATCH = {")": "(", "]": "[", "}": "{"}


def _is_valid(s: str) -> bool:
    stack = []
    for ch in s:
        if ch in MATCH:
            if not stack or stack[-1] != MATCH[ch]:
                return False
            stack.pop()
        else:
            stack.append(ch)
    return not stack


def _large_case():
    # 10^4 characters, the constraint limit: 5000 pairs in random nesting. Near the end, two
    # adjacent closing brackets of different types are swapped. The bracket counts stay
    # balanced, so a count-only check says true, but the answer is false. The "delete () [] {}
    # until nothing changes" approach still finishes quickly (str.replace runs in C)
    rng = random.Random(20)
    opens, stack, out = 5000, [], []
    while opens or stack:
        if opens and (not stack or rng.random() < 0.5):
            o = rng.choice("([{")
            out.append(o)
            stack.append(o)
            opens -= 1
        else:
            out.append({"(": ")", "[": "]", "{": "}"}[stack.pop()])
    i = max(j for j in range(len(out) - 1) if out[j] in ")]}" and out[j + 1] in ")]}" and out[j] != out[j + 1])
    out[i], out[i + 1] = out[i + 1], out[i]
    s = "".join(out)
    return {"args": [s], "expected": _is_valid(s)}


def _random_case(rng):
    kinds = [("(", ")"), ("[", "]"), ("{", "}")]
    pairs = rng.randint(1, 4)
    if rng.random() < 0.5:
        # Valid nesting of random bracket types
        out, stack, opens = [], [], pairs
        while opens or stack:
            if opens and (not stack or rng.random() < 0.5):
                o, c = rng.choice(kinds)
                out.append(o)
                stack.append(c)
                opens -= 1
            else:
                out.append(stack.pop())
        return {"args": ["".join(out)]}
    # The same bracket counts in a shuffled order: usually invalid, with balanced counts
    chars = []
    for _ in range(pairs):
        o, c = rng.choice(kinds)
        chars += [o, c]
    rng.shuffle(chars)
    return {"args": ["".join(chars)]}


TASK = {
    "title": "Valid Parentheses",
    "title_zh": "有效的括号",
    "difficulty": "Easy",
    "leetcode_id": 20,
    "leetcode_slug": "valid-parentheses",
    "category": "Stack",
    "function_name": "isValid",
    "params": [("s", "str")],
    "returns": "bool",
    "description_en": "Given a string `s` made only of the characters `(`, `)`, `[`, `]`, `{` and `}`, decide whether it is valid.\n\nA string is valid when every opening bracket is closed by a bracket of the same type, the closing brackets come in the right order, and every closing bracket has a matching opening bracket before it.\n\n**Example 1:** `s = \"[{}]()\"` → `true`\n\n**Example 2:** `s = \"([)]\"` → `false`\n\n**Constraints:**\n- $1 \\le \\texttt{s.length} \\le 10^{4}$\n- `s` consists of the bracket characters `()[]{}` only",
    "description_zh": "给定一个只由字符 `(`、`)`、`[`、`]`、`{`、`}` 组成的字符串 `s`，判断它是否有效。\n\n有效的条件是：每个左括号都必须由同类型的右括号闭合，括号闭合的顺序必须正确，并且每个右括号前面都有与之匹配的左括号。\n\n**示例 1：** `s = \"[{}]()\"` → `true`\n\n**示例 2：** `s = \"([)]\"` → `false`\n\n**约束：**\n- $1 \\le \\texttt{s.length} \\le 10^{4}$\n- `s` 只包含括号字符 `()[]{}`",
    "hint": "Keep the opening brackets you still have to close on a stack. Each closing bracket must match the top of the stack, and at the end the stack must be empty.",
    "hint_zh": "用栈保存尚未闭合的左括号。每个右括号都必须与栈顶匹配，结束时栈必须为空。",
    "cases": [
        {"args": ["[{}]()"], "expected": True},
        {"args": ["([)]"], "expected": False},
        {"args": ["("], "expected": False},
        # Closing before opening: the counts match, but the order is wrong
        {"args": [")("], "expected": False},
        {"args": ["{[()]}"], "expected": True},
        {"args": ["(("], "expected": False},
        # A lone closing bracket finds an empty stack: check before popping
        {"args": ["]"], "expected": False},
        {"args": ["()[]{}"], "expected": True},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def isValid(self, s: str) -> bool:
        match = {")": "(", "]": "[", "}": "{"}
        stack = []  # opening brackets still waiting for their closer
        for ch in s:
            if ch in match:
                if not stack or stack[-1] != match[ch]:
                    return False
                stack.pop()
            else:
                stack.append(ch)
        return not stack''',
    "explanation": "Scan left to right with a stack of unclosed opening brackets. An opening bracket is pushed. A closing bracket must match the top of the stack, which is then popped; if the stack is empty or the top is a different type, the string is invalid. At the end, any leftover opening bracket makes it invalid too.\n\nTime O(n), space O(n).",
    "explanation_zh": "从左到右扫描，用栈保存尚未闭合的左括号。遇到左括号就入栈。遇到右括号时，它必须与栈顶匹配，然后弹出栈顶；如果栈为空或栈顶类型不同，字符串无效。扫描结束后，如果栈中还剩左括号，同样无效。\n\n时间 O(n)，空间 O(n)。",
}
