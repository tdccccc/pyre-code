"""Letter Combinations of a Phone Number (LeetCode 17)."""

import itertools
import random

KEYS = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}


def _large_case():
    # At the constraint limit: 4 digits, each on a 4-letter key, so 4^4 = 256 strings
    rng = random.Random(17)
    digits = "".join(rng.choice("79") for _ in range(4))
    expected = ["".join(p) for p in itertools.product(*(KEYS[d] for d in digits))]
    return {"args": [digits], "expected": expected}


def _random_case(rng):
    # Empty input is common here on purpose: it is the case a careless starting value gets wrong
    return {"args": ["".join(rng.choice("23456789") for _ in range(rng.randint(0, 3)))]}


TASK = {
    "title": "Letter Combinations of a Phone Number",
    "title_zh": "电话号码的字母组合",
    "difficulty": "Medium",
    "leetcode_id": 17,
    "leetcode_slug": "letter-combinations-of-a-phone-number",
    "category": "Backtracking",
    "function_name": "letterCombinations",
    "params": [("digits", "str")],
    "returns": "str[]",
    "check": "unordered",
    "description_en": """Each digit from `2` to `9` stands for a key on an old phone keypad, and each key carries a few letters:

`2` → abc, `3` → def, `4` → ghi, `5` → jkl, `6` → mno, `7` → pqrs, `8` → tuv, `9` → wxyz

Given a string `digits`, return every string of letters that can be typed by pressing one key per digit, taking one of that key's letters each time. For example, `4` followed by `2` can produce `ga`, `gb`, and so on up to `ic`. The strings may be returned in any order. If `digits` is empty, return an empty list.

**Example 1:** `digits = "42"` → `["ga","gb","gc","ha","hb","hc","ia","ib","ic"]`

**Example 2:** `digits = ""` → `[]`

**Constraints:**
- $0 \\le \\texttt{digits.length} \\le 4$
- `digits[i]` is a digit from `2` to `9`""",
    "description_zh": """数字 `2` 到 `9` 各自对应老式电话键盘上的一个按键，每个按键上有若干字母：

`2` → abc，`3` → def，`4` → ghi，`5` → jkl，`6` → mno，`7` → pqrs，`8` → tuv，`9` → wxyz

给定字符串 `digits`，返回依次按下这些按键、每次取该键上一个字母所能拼出的全部字母串。例如先按 `4` 再按 `2`，可以拼出 `ga`、`gb`，一直到 `ic`。结果的顺序任意。若 `digits` 为空，返回空列表。

**示例 1：** `digits = "42"` → `["ga","gb","gc","ha","hb","hc","ia","ib","ic"]`

**示例 2：** `digits = ""` → `[]`

**约束：**
- $0 \\le \\texttt{digits.length} \\le 4$
- `digits[i]` 是 `2` 到 `9` 之间的数字""",
    "hint": "Extend the strings one digit at a time: for each string built so far, append each letter of the next digit's key. Think about what the starting value should be so that an empty input gives an empty list.",
    "hint_zh": "每次处理一个数字：对已有的每个字符串，逐个追加下一个数字按键上的字母。想一想起始值应该是什么，才能让空输入得到空列表。",
    "cases": [
        {"args": ["42"], "expected": ["ga", "gb", "gc", "ha", "hb", "hc", "ia", "ib", "ic"]},
        # Empty input gives an empty list, not a list holding the empty string
        {"args": [""], "expected": []},
        {"args": ["6"], "expected": ["m", "n", "o"]},
        {"args": ["7"], "expected": ["p", "q", "r", "s"]},
        {"args": ["25"], "expected": ["aj", "ak", "al", "bj", "bk", "bl", "cj", "ck", "cl"]},
        {"args": ["79"], "expected": ["pw", "px", "py", "pz", "qw", "qx", "qy", "qz", "rw", "rx", "ry", "rz", "sw", "sx", "sy", "sz"]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        keys = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl",
                "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        combos = [""]
        for d in digits:
            combos = [prefix + ch for prefix in combos for ch in keys[d]]
        return combos''',
    "explanation": "Build the strings one digit at a time. Keep the list of partial strings; for the next digit, replace each partial string with one copy per letter on that digit's key. The empty input is a special case: there is nothing to type, so the answer is an empty list rather than the list holding the empty string.\n\nTime O(4^n · n) for n digits (the output size), space O(4^n · n) for the output.",
    "explanation_zh": "逐位构造字符串。保存当前所有的部分字符串；处理下一个数字时，把每个部分字符串替换为「加上该键上每个字母」的若干个新字符串。空输入是特例：没有可按的键，答案是空列表，而不是包含空字符串的列表。\n\n时间 O(4^n · n)（n 为位数，即输出规模），空间同为输出规模。",
}
