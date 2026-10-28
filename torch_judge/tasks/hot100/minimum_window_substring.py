"""Minimum Window Substring (LeetCode 76)."""

import random
import string
from collections import Counter


def _min_window(s: str, t: str) -> str:
    # Independent method: count letters inside the window and track how many kinds are satisfied
    want = Counter(t)
    have = Counter()
    kinds = len(want)
    satisfied = 0
    best = (float("inf"), 0, 0)
    left = 0
    for right, ch in enumerate(s):
        have[ch] += 1
        if ch in want and have[ch] == want[ch]:
            satisfied += 1
        while satisfied == kinds:
            if right - left + 1 < best[0]:
                best = (right - left + 1, left, right + 1)
            out = s[left]
            if out in want and have[out] == want[out]:
                satisfied -= 1
            have[out] -= 1
            left += 1
    return s[best[1]:best[2]]


def _large_case():
    # 10⁵ letters (both cases) and a 4000-letter target. Checking each start with a full recount
    # costs about 4·10⁸ steps; the sliding window moves each pointer across the text once.
    rng = random.Random(76)
    alphabet = string.ascii_letters
    s = "".join(rng.choice(alphabet) for _ in range(100000))
    t = "".join(rng.choice(alphabet) for _ in range(4000))
    return {"args": [s, t], "expected": _min_window(s, t)}


def _random_case(rng):
    # Three letters: windows overlap a lot, and t often repeats a letter
    s = "".join(rng.choice("abc") for _ in range(rng.randint(1, 12)))
    t = "".join(rng.choice("abc") for _ in range(rng.randint(1, 4)))
    return {"args": [s, t]}


TASK = {
    "title": "Minimum Window Substring",
    "title_zh": "最小覆盖子串",
    "difficulty": "Hard",
    "leetcode_id": 76,
    "leetcode_slug": "minimum-window-substring",
    "category": "Substring",
    "function_name": "minWindow",
    "params": [("s", "str"), ("t", "str")],
    "returns": "str",
    # Several windows can tie for the shortest, so any shortest valid window is accepted
    "check": "custom",
    "checker": '''def check(args, actual, expected):
    from collections import Counter
    s, t = args
    if expected == "":
        return actual == ""
    return (isinstance(actual, str) and len(actual) == len(expected) and actual in s
            and not Counter(t) - Counter(actual))''',
    "description_en": "Given strings `s` and `t`, return the shortest substring of `s` that contains every letter of `t`, counting repeats: each letter must appear at least as many times as it does in `t`. Letter case matters. If no such substring exists, return the empty string `\"\"`. If several substrings tie for the shortest length, return any of them.\n\n**Example 1:** `s = \"xaybzcz\", t = \"abc\"` → `\"aybzc\"`\n\n**Example 2:** `s = \"ab\", t = \"bb\"` → `\"\"` (the window would need two `b`s)\n\n**Constraints:**\n- $1 \\le \\texttt{s.length},\\ \\texttt{t.length} \\le 10^{5}$\n- `s` and `t` consist of English letters",
    "description_zh": "给定字符串 `s` 和 `t`，返回 `s` 中最短的子串，要求它包含 `t` 中的每个字符，且每个字符出现的次数不少于它在 `t` 中出现的次数。字母区分大小写。若不存在这样的子串，返回空字符串 `\"\"`。若有多个同样最短的子串，返回其中任意一个。\n\n**示例 1：** `s = \"xaybzcz\", t = \"abc\"` → `\"aybzc\"`\n\n**示例 2：** `s = \"ab\", t = \"bb\"` → `\"\"`（窗口里需要两个 `b`）\n\n**约束：**\n- $1 \\le \\texttt{s.length},\\ \\texttt{t.length} \\le 10^{5}$\n- `s` 和 `t` 由英文字母组成",
    "hint": "Slide a window over `s` and track how many letters of `t` are still owed. Once the window covers everything, shrink it from the left as far as possible, recording the best length each time.",
    "hint_zh": "在 s 上滑动窗口，记录 t 中仍然缺少的字符数。窗口覆盖全部所需字符后，尽量从左侧收缩，每次都记录最短长度。",
    "cases": [
        {"args": ["xaybzcz", "abc"], "expected": "aybzc"},
        {"args": ["ab", "bb"], "expected": ""},
        {"args": ["a", "a"], "expected": "a"},
        # t needs two a's; a presence-only check would wrongly return "a"
        {"args": ["a", "aa"], "expected": ""},
        {"args": ["ooxabxcxxx", "abc"], "expected": "abxc"},
        # Repeats matter: the window needs two a's and one b. "ab" has every letter but only one a
        {"args": ["abbaac", "aab"], "expected": "baa"},
        # Case-sensitive: the window must contain "A", and "a" does not count
        {"args": ["aA", "A"], "expected": "A"},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''from collections import Counter


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = Counter(t)  # letters the window still owes (negative: the window has a surplus)
        missing = len(t)  # total letters still owed
        best_start, best_len = 0, len(s) + 1
        left = 0
        for right, ch in enumerate(s):
            if need[ch] > 0:
                missing -= 1
            need[ch] -= 1
            if missing == 0:
                # Shrink past surplus letters only, so the window still covers t
                while need[s[left]] < 0:
                    need[s[left]] += 1
                    left += 1
                if right - left + 1 < best_len:
                    best_start, best_len = left, right - left + 1
                # Give up the leftmost letter (it was exactly needed) and keep looking
                need[s[left]] += 1
                missing += 1
                left += 1
        return s[best_start:best_start + best_len] if best_len <= len(s) else ""''',
    "explanation": "Two pointers over `s`. `need[c]` is how many `c` the window still owes, negative when it holds a surplus, and `missing` is the sum of what is still owed. Extend the right pointer until `missing` is 0. Then shrink from the left past surplus letters, record the window, and release the leftmost required letter to search for a shorter one. Each pointer crosses the string once.\n\nTime O(n + m), space O(alphabet size).",
    "explanation_zh": "对 `s` 使用双指针。`need[c]` 表示窗口还欠 `c` 的个数（负数表示多余），`missing` 是仍然欠缺的字符总数。右指针扩展直到 `missing` 为 0；然后左指针跨过多余的字符向右收缩，记录当前窗口，再放掉最左边的必需字符，继续寻找更短的窗口。两个指针各只扫过字符串一遍。\n\n时间 O(n + m)，空间 O(字符集大小)。",
}
