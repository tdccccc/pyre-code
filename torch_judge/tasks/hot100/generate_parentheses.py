"""Generate Parentheses (LeetCode 22)."""

import itertools


def _is_well_formed(s: str) -> bool:
    depth = 0
    for ch in s:
        depth += 1 if ch == "(" else -1
        if depth < 0:
            return False
    return depth == 0


def _large_case():
    # At the constraint limit: 8 pairs give Catalan(8) = 1430 answers. Checking all 2^16
    # bracket strings is fast enough to serve as the expected answer.
    n = 8
    expected = ["".join(p) for p in itertools.product("()", repeat=2 * n) if _is_well_formed("".join(p))]
    return {"args": [n], "expected": expected}


def _random_case(rng):
    return {"args": [rng.randint(1, 4)]}


TASK = {
    "title": "Generate Parentheses",
    "title_zh": "括号生成",
    "difficulty": "Medium",
    "leetcode_id": 22,
    "leetcode_slug": "generate-parentheses",
    "category": "Backtracking",
    "function_name": "generateParenthesis",
    "params": [("n", "int")],
    "returns": "str[]",
    "check": "unordered",
    "description_en": """Given an integer `n`, write out every well-formed string made of `n` opening brackets `(` and `n` closing brackets `)`. A string is well-formed when, read from left to right, it never has more `)` than `(` so far, and the two counts are equal at the end. Return the strings in any order.

**Example 1:** `n = 2` → `["(())","()()"]`

**Example 2:** `n = 1` → `["()"]`

**Constraints:**
- `1 <= n <= 8`""",
    "description_zh": """给定整数 `n`，写出所有由 `n` 个左括号 `(` 和 `n` 个右括号 `)` 组成的合法字符串。合法指从左到右读时，任何位置的 `)` 数量都不超过 `(` 的数量，并且最终两者数量相等。返回的字符串顺序任意。

**示例 1：** `n = 2` → `["(())","()()"]`

**示例 2：** `n = 1` → `["()"]`

**约束：**
- `1 <= n <= 8`""",
    "hint": "Build the string left to right and only add a `)` when there is an unmatched `(` before it. Track how many of each you have used.",
    "hint_zh": "从左到右构造字符串，只有在前面还有未匹配的 `(` 时才能添加 `)`。记录两种括号各用了多少。",
    "cases": [
        {"args": [2], "expected": ["(())", "()()"]},
        {"args": [1], "expected": ["()"]},
        {"args": [3], "expected": ["((()))", "(()())", "(())()", "()(())", "()()()"]},
        # Four pairs: the full set of 14 well-formed strings
        {"args": [4], "expected": ["(((())))", "((()()))", "((())())", "((()))()", "(()(()))", "(()()())",
                                   "(()())()", "(())(())", "(())()()", "()((()))", "()(()())", "()(())()",
                                   "()()(())", "()()()()"]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        result = []

        def backtrack(current: str, opened: int, closed: int) -> None:
            if len(current) == 2 * n:
                result.append(current)
                return
            if opened < n:
                backtrack(current + "(", opened + 1, closed)
            if closed < opened:  # a ")" is legal only if it closes an earlier "("
                backtrack(current + ")", opened, closed + 1)

        backtrack("", 0, 0)
        return result''',
    "explanation": "Backtracking that builds the string left to right with two counters: opening brackets used and closing brackets used. An opening bracket may be added while fewer than n are used. A closing bracket may be added only while it does not outnumber the opening ones, which is exactly what keeps every prefix valid. Every complete string produced this way is well-formed, and every well-formed string is reached.\n\nTime O(4^n / √n) (proportional to the output size, Catalan(n) strings of length 2n), space O(n) for the recursion.",
    "explanation_zh": "回溯法，从左到右构造字符串，同时记录已用的左括号数和右括号数。左括号只要还没用满 n 个就可以加；右括号只有在它不会超过左括号数量时才能加，这一条正好保证每个前缀都合法。这样生成的每个完整字符串都合法，并且每个合法字符串都会被生成。\n\n时间与输出规模成正比（共 Catalan(n) 个字符串，每个长 2n），递归空间 O(n)。",
}
