"""Group Anagrams (LeetCode 49)."""

import random
from collections import defaultdict


def _large_case():
    rng = random.Random(49)
    bases = ["".join(rng.choice("abcdefghij") for _ in range(8)) for _ in range(2000)]
    strs = []
    for _ in range(20000):
        chars = list(rng.choice(bases))
        rng.shuffle(chars)
        strs.append("".join(chars))
    groups = defaultdict(list)
    for s in strs:
        groups["".join(sorted(s))].append(s)
    return {"args": [strs], "expected": list(groups.values())}


def _random_case(rng):
    # A tiny alphabet and short words make anagram collisions (and exact duplicates) common
    words = []
    for _ in range(rng.randint(1, 8)):
        length = rng.randint(1, 3)
        words.append("".join(rng.choice("abc") for _ in range(length)))
    return {"args": [words]}


TASK = {
    "title": "Group Anagrams",
    "title_zh": "字母异位词分组",
    "difficulty": "Medium",
    "leetcode_id": 49,
    "leetcode_slug": "group-anagrams",
    "category": "Hashing",
    "function_name": "groupAnagrams",
    "params": [("strs", "str[]")],
    "returns": "str[][]",
    "check": "unordered_nested",
    "description_en": "Given a list of strings `strs`, put strings that are anagrams of each other (same letters, same counts, any order) into the same group, and return the list of groups.\n\nGroups may appear in any order, and so may the strings inside a group.\n\n**Example 1:** `strs = [\"stone\",\"notes\",\"tab\",\"onset\",\"bat\",\"cat\"]` → `[[\"stone\",\"notes\",\"onset\"],[\"tab\",\"bat\"],[\"cat\"]]`\n\n**Example 2:** `strs = [\"a\"]` → `[[\"a\"]]`\n\n**Constraints:**\n- `1 <= strs.length <= 10^4`\n- `0 <= strs[i].length <= 100`\n- `strs[i]` contains lowercase English letters only",
    "description_zh": "给定字符串列表 `strs`，把互为字母异位词（字母相同、各字母出现次数相同、顺序任意）的字符串放进同一组，返回所有分组。\n\n组的顺序、组内字符串的顺序都不限。\n\n**示例 1：** `strs = [\"stone\",\"notes\",\"tab\",\"onset\",\"bat\",\"cat\"]` → `[[\"stone\",\"notes\",\"onset\"],[\"tab\",\"bat\"],[\"cat\"]]`\n\n**示例 2：** `strs = [\"a\"]` → `[[\"a\"]]`\n\n**约束：**\n- `1 <= strs.length <= 10^4`\n- `0 <= strs[i].length <= 100`\n- `strs[i]` 只包含小写英文字母",
    "hint": "Anagrams share a canonical form, e.g. the sorted string or a 26-letter count tuple. Use it as a dict key.",
    "hint_zh": "字母异位词有相同的「标准形式」，比如排序后的字符串，或 26 个字母的计数元组。用它当字典的键。",
    "cases": [
        {"args": [["stone", "notes", "tab", "onset", "bat", "cat"]], "expected": [["stone", "notes", "onset"], ["tab", "bat"], ["cat"]]},
        {"args": [["a"]], "expected": [["a"]]},
        # An empty string prints as an empty line, which ACM output can't tell from "no group"
        {"args": [[""]], "expected": [[""]], "acm": False},
        {"args": [["ab", "ba", "abc", "cab", "bca", "xyz"]], "expected": [["ab", "ba"], ["abc", "cab", "bca"], ["xyz"]]},
        {"args": [["aab", "aba", "abb"]], "expected": [["aab", "aba"], ["abb"]]},
        # Exact duplicates stay as separate entries in the same group
        {"args": [["a", "a", "b"]], "expected": [["a", "a"], ["b"]]},
        # Same letter multiset, different lengths: not anagrams
        {"args": [["ab", "aab", "abb", "ba"]], "expected": [["ab", "ba"], ["aab"], ["abb"]]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''from collections import defaultdict


class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groups = defaultdict(list)
        for s in strs:
            groups["".join(sorted(s))].append(s)
        return list(groups.values())''',
    "explanation": "Map each string to a key that is identical for all of its anagrams; sorting the characters works. Collect strings by key in a dict and return the dict's values.\n\nWith n strings of length up to k: time O(n·k log k), space O(n·k). A 26-count tuple key brings the time down to O(n·k).",
    "explanation_zh": "给每个字符串算一个「所有异位词都相同」的键，把字符排序就行。用字典按键收集字符串，最后返回字典的所有值。\n\nn 个字符串、长度不超过 k：时间 O(n·k log k)，空间 O(n·k)。改用 26 个字母的计数元组当键，时间可以降到 O(n·k)。",
}
