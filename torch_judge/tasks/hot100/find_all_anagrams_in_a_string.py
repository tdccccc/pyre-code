"""Find All Anagrams in a String (LeetCode 438)."""

import random
import string
from collections import Counter


def _anagram_starts(s: str, p: str) -> list[int]:
    # Fixed-size window, recounted with Counters: each step adds one letter and drops one
    m = len(p)
    want = Counter(p)
    have = Counter(s[:m])
    out = [0] if have == want else []
    for i in range(m, len(s)):
        have[s[i]] += 1
        have[s[i - m]] -= 1
        if have[s[i - m]] == 0:
            del have[s[i - m]]
        if have == want:
            out.append(i - m + 1)
    return out


def _large_case():
    # Text of 3·10⁴ letters and a pattern of 10⁴ letters over "abc", with three shuffled copies of
    # the pattern planted in the text. Sorting or counting every window costs about 2·10⁸ steps.
    rng = random.Random(438)
    p = [rng.choice("abc") for _ in range(10000)]
    s = [rng.choice("abc") for _ in range(30000)]
    for start in rng.sample(range(0, 20000), 3):
        copy = p[:]
        rng.shuffle(copy)
        s[start:start + len(copy)] = copy
    s, p = "".join(s), "".join(p)
    return {"args": [s, p], "expected": _anagram_starts(s, p)}


def _random_case(rng):
    # Two letters, so a window often has the right letters in the wrong counts (p = "aab"
    # against window "abb"), which the mistake misses. Measured: about 10 of 20 cases expose it
    s = "".join(rng.choice("ab") for _ in range(rng.randint(6, 12)))
    p = "".join(rng.choice("ab") for _ in range(rng.randint(3, 4)))
    return {"args": [s, p]}


TASK = {
    "title": "Find All Anagrams in a String",
    "title_zh": "找到字符串中所有字母异位词",
    "difficulty": "Medium",
    "leetcode_id": 438,
    "leetcode_slug": "find-all-anagrams-in-a-string",
    "category": "Sliding Window",
    "function_name": "findAnagrams",
    "params": [("s", "str"), ("p", "str")],
    "returns": "int[]",
    "check": "unordered",
    "description_en": "Given two strings `s` and `p`, return the starting index of every substring of `s` that is an anagram of `p`. An anagram here means the same letters with the same counts, in any order. The indices may be returned in any order.\n\n**Example 1:** `s = \"abcabcab\", p = \"abc\"` → `[0,1,2,3,4,5]`\n\n**Example 2:** `s = \"xyzzy\", p = \"zy\"` → `[1,3]` (the substrings `yz` and `zy`)\n\n**Constraints:**\n- `1 <= p.length, s.length <= 3 * 10^4`\n- `s` and `p` consist of lowercase English letters",
    "description_zh": "给定两个字符串 `s` 和 `p`，返回 `s` 中每个是 `p` 的字母异位词的子串的起始下标。这里的异位词指字母种类和每种字母的个数都相同、只是顺序不同。下标可以按任意顺序返回。\n\n**示例 1：** `s = \"abcabcab\", p = \"abc\"` → `[0,1,2,3,4,5]`\n\n**示例 2：** `s = \"xyzzy\", p = \"zy\"` → `[1,3]`（子串 `yz` 和 `zy`）\n\n**约束：**\n- `1 <= p.length, s.length <= 3 * 10^4`\n- `s` 和 `p` 只由小写英文字母组成",
    "hint": "Keep a window of length `len(p)` and move it one character at a time. Track how far the window's letter counts are from p's counts; the window is an anagram when every difference is zero.",
    "hint_zh": "维护一个长度为 `len(p)` 的窗口，每次只增减一个字符。记录窗口字母计数与 p 的差值，所有差值都为 0 时即为异位词。",
    "cases": [
        {"args": ["abcabcab", "abc"], "expected": [0, 1, 2, 3, 4, 5]},
        {"args": ["xyzzy", "zy"], "expected": [1, 3]},
        {"args": ["ab", "abc"], "expected": []},
        # Same letter set, different counts: a set-based comparison wrongly accepts windows 0 and 1
        {"args": ["abbaab", "aab"], "expected": [2, 3]},
        {"args": ["baa", "aa"], "expected": [1]},
        {"args": ["aaaaa", "a"], "expected": [0, 1, 2, 3, 4]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''from collections import Counter


class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        m = len(p)
        # balance[c] = count of c in the window minus count of c in p
        balance = Counter()
        for ch in p:
            balance[ch] -= 1
        nonzero = sum(1 for v in balance.values() if v)  # letters whose balance is off
        res = []
        for i, ch in enumerate(s):
            before = balance[ch]
            balance[ch] += 1  # s[i] enters the window
            nonzero += (balance[ch] != 0) - (before != 0)
            if i >= m:
                out = s[i - m]  # s[i - m] leaves it
                before = balance[out]
                balance[out] -= 1
                nonzero += (balance[out] != 0) - (before != 0)
            if i >= m - 1 and nonzero == 0:
                res.append(i - m + 1)
        return res''',
    "explanation": "A fixed-size sliding window. `balance[c]` holds the count of `c` in the window minus its count in `p`, and `nonzero` counts the letters whose balance is not zero. Adding one letter and dropping another changes at most two balances, so `nonzero` is updated in O(1). The window is an anagram exactly when `nonzero` is 0.\n\nTime O(n + m), space O(alphabet size).",
    "explanation_zh": "定长滑动窗口。`balance[c]` 记录窗口中字符 `c` 的个数减去 `p` 中的个数，`nonzero` 统计 balance 不为零的字母种数。每次加入一个字符、移出一个字符最多只改变两个 balance，所以 `nonzero` 可以 O(1) 更新。`nonzero` 为 0 时窗口就是异位词。\n\n时间 O(n + m)，空间 O(字符集大小)。",
}
