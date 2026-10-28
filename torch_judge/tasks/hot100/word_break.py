"""Word Break (LeetCode 139)."""

import random


def _can_split(s: str, words: list[str]) -> bool:
    # Forward reachability over prefixes of s: reach[i] means s[:i] can be split
    reach = [True] + [False] * len(s)
    for i in range(1, len(s) + 1):
        reach[i] = any(reach[j] and s[j:i] in words for j in range(i))
    return reach[-1]


def _large_case():
    # s = 299 'a's and a final 'b', with words a, aa, ..., a*19. The answer is false because of
    # the 'b', but a search that tries every split of the a's (about 2^298 ways) never finishes
    rng = random.Random(139)
    words = ["a" * k for k in range(1, 20)]
    rng.shuffle(words)
    s = "a" * 299 + "b"
    return {"args": [s, words], "expected": _can_split(s, words)}


def _random_case(rng):
    # Words and strings over {a, b}, so the dictionary splits the string in more than one way
    # and greedy choices go wrong. Half the strings are built from the words (true cases)
    words = sorted({"".join(rng.choice("ab") for _ in range(rng.randint(1, 3))) for _ in range(rng.randint(1, 4))})
    if rng.random() < 0.5:
        s = "".join(rng.choice(words) for _ in range(rng.randint(1, 4)))
    else:
        s = "".join(rng.choice("ab") for _ in range(rng.randint(1, 8)))
    return {"args": [s, words]}


TASK = {
    "title": "Word Break",
    "title_zh": "单词拆分",
    "difficulty": "Medium",
    "leetcode_id": 139,
    "leetcode_slug": "word-break",
    "category": "Dynamic Programming",
    "function_name": "wordBreak",
    "params": [("s", "str"), ("wordDict", "str[]")],
    "returns": "bool",
    "description_en": "Given a string `s` and a list of words `wordDict`, decide whether `s` can be cut into pieces that each appear in `wordDict`. The pieces must cover all of `s` with no gaps or overlaps, and a word may be used as many times as needed.\n\nReturn `true` if such a cut exists, otherwise `false`.\n\n**Example 1:** `s = \"nightfall\", wordDict = [\"night\",\"fall\",\"nigh\"]` → `true` (`night` + `fall`)\n\n**Example 2:** `s = \"abcef\", wordDict = [\"ab\",\"abc\",\"cd\"]` → `false`\n\n**Constraints:**\n- `1 <= s.length <= 300`\n- `1 <= wordDict.length <= 50`\n- `1 <= wordDict[i].length <= 20`\n- `s` and every `wordDict[i]` consist of lowercase English letters",
    "description_zh": "给定字符串 `s` 和单词列表 `wordDict`，判断 `s` 能否切分成若干片段，使每个片段都出现在 `wordDict` 中。片段必须恰好拼成整个 `s`，不能遗漏或重叠；同一个单词可以重复使用。\n\n能切分则返回 `true`，否则返回 `false`。\n\n**示例 1：** `s = \"nightfall\", wordDict = [\"night\",\"fall\",\"nigh\"]` → `true`（`night` + `fall`）\n\n**示例 2：** `s = \"abcef\", wordDict = [\"ab\",\"abc\",\"cd\"]` → `false`\n\n**约束：**\n- `1 <= s.length <= 300`\n- `1 <= wordDict.length <= 50`\n- `1 <= wordDict[i].length <= 20`\n- `s` 与每个 `wordDict[i]` 都只由小写英文字母组成",
    "hint": "Let ok[i] mean that the prefix s[:i] can be split. Then ok[i] is true when some ok[j] is true and s[j:i] is a word. Only j within 20 characters of i needs checking.",
    "hint_zh": "设 ok[i] 表示前缀 s[:i] 能否切分。若存在 ok[j] 为真且 s[j:i] 是一个单词，则 ok[i] 为真。只需检查与 i 相距不超过 20 个字符的 j。",
    "cases": [
        {"args": ["nightfall", ["night", "fall", "nigh"]], "expected": True},
        {"args": ["abcef", ["ab", "abc", "cd"]], "expected": False},
        # Greedy longest-match takes "abc", strands the "d", and fails. "ab" + "cd" works
        {"args": ["abcd", ["ab", "abc", "cd"]], "expected": True},
        # Greedy longest-match takes "aba", then "bab" has no match; "ab" x 3 works
        {"args": ["ababab", ["ab", "aba"]], "expected": True},
        {"args": ["a", ["b"]], "expected": False},
        {"args": ["bb", ["b"]], "expected": True},
        {"args": ["aaaaab", ["a", "aa"]], "expected": False},
        {"args": ["leetcodex", ["leet", "code", "lee", "tcode"]], "expected": False},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        words = set(wordDict)
        longest = max(map(len, words))
        ok = [True] + [False] * len(s)  # ok[i]: s[:i] can be split into words
        for i in range(1, len(s) + 1):
            ok[i] = any(ok[j] and s[j:i] in words for j in range(max(0, i - longest), i))
        return ok[-1]''',
    "explanation": "`ok[i]` is true when the prefix `s[:i]` can be split into dictionary words. The last word in such a split is `s[j:i]` for some `j`, and the part before it must itself be splittable, so `ok[i] = any(ok[j] and s[j:i] in words)`. A word has at most `L` letters, where `L` is the longest word, so only the `L` values of `j` before `i` need checking.\n\nTime O(n·L²) (a slice and a hash lookup for each check), space O(n + total word length).",
    "explanation_zh": "`ok[i]` 为真，表示前缀 `s[:i]` 能被拆成词典中的单词。这样的拆分中，最后一个单词是 `s[j:i]`（某个 `j`），而它之前的部分也必须能拆分，因此 `ok[i] = any(ok[j] and s[j:i] in words)`。单词最长为 `L` 个字母，所以只需检查 `i` 之前的 `L` 个 `j`。\n\n时间 O(n·L²)（每次检查包含一次切片和一次哈希查找），空间 O(n + 单词总长)。",
}
