"""Implement Trie (Prefix Tree) (LeetCode 208)."""

import bisect
import random


def _has_prefix(sorted_words: list, prefix: str) -> bool:
    # Words that share a prefix are adjacent in sorted order, so the first word >= prefix decides
    i = bisect.bisect_left(sorted_words, prefix)
    return i < len(sorted_words) and sorted_words[i].startswith(prefix)


def _large_case():
    # 3 * 10^4 calls, about a third of them inserts, on words of up to 40 letters. Each call
    # should cost about the length of its word. The expected answers come from a set and a
    # sorted word list, not from a trie.
    rng = random.Random(208)
    letters = "abcdefghijklmnopqrstuvwxyz"
    ops, args, expected = ["Trie"], [[]], [None]
    words, sorted_words, stored = [], [], set()
    for _ in range(30000):
        if not words or rng.random() < 0.35:
            word = "".join(rng.choice(letters) for _ in range(rng.randint(1, 40)))
            ops.append("insert")
            args.append([word])
            expected.append(None)
            if word not in stored:
                stored.add(word)
                words.append(word)
                bisect.insort(sorted_words, word)
            continue
        base = rng.choice(words)
        if rng.random() < 0.5:
            word = base if rng.random() < 0.5 else base[: rng.randint(1, len(base))]
            ops.append("search")
            args.append([word])
            expected.append(word in stored)
        else:
            prefix = base[: rng.randint(1, len(base))]
            if rng.random() < 0.3:
                prefix += rng.choice(letters)
            ops.append("startsWith")
            args.append([prefix])
            expected.append(_has_prefix(sorted_words, prefix))
    return {"ops": ops, "args": args, "expected": expected}


def _random_case(rng):
    # Words over "ab" are short, so prefixes, repeats and near misses come up often
    ops, args, words = ["Trie"], [[]], []
    for _ in range(rng.randint(6, 12)):
        if not words or rng.random() < 0.4:
            word = "".join(rng.choice("ab") for _ in range(rng.randint(1, 3)))
            words.append(word)
            ops.append("insert")
            args.append([word])
        else:
            base = rng.choice(words)
            # a prefix of a stored word, or that word with one more letter
            query = base[: rng.randint(1, len(base))] if rng.random() < 0.6 else base + rng.choice("ab")
            ops.append(rng.choice(["search", "startsWith"]))
            args.append([query])
    return {"ops": ops, "args": args}


TASK = {
    "title": "Implement Trie (Prefix Tree)",
    "title_zh": "实现 Trie (前缀树)",
    "difficulty": "Medium",
    "leetcode_id": 208,
    "leetcode_slug": "implement-trie-prefix-tree",
    "category": "Graphs",
    "function_name": "Trie",
    "design": {
        "class": "Trie",
        "init": {},
        "methods": {
            "insert": ({"word": "str"}, "void"),
            "search": ({"word": "str"}, "bool"),
            "startsWith": ({"prefix": "str"}, "bool"),
        },
    },
    "description_en": """Build a dictionary of lowercase words that can also answer prefix questions. Implement the class `Trie`:

- `Trie()` creates an empty dictionary
- `insert(word)` adds `word` to the dictionary
- `search(word)` returns `true` if exactly `word` was inserted, and `false` otherwise (a shorter or longer string does not count)
- `startsWith(prefix)` returns `true` if at least one inserted word begins with `prefix`

**Example 1:** `insert("grape")`, `insert("gram")`, `search("gram")` → `true`, `search("gra")` → `false`, `startsWith("gra")` → `true`, `startsWith("pear")` → `false`

**Example 2:** `insert("car")`, `search("card")` → `false`, `startsWith("ca")` → `true`, `insert("card")`, `search("card")` → `true`

**Constraints:**
- $1 \\le \\texttt{word.length},\\ \\texttt{prefix.length} \\le 40$
- `word` and `prefix` consist of lowercase English letters only
- At most $3 \\times 10^{4}$ calls in total to `insert`, `search` and `startsWith`

Each call should take time proportional to the length of its string, not to the number of stored words.""",
    "description_zh": """设计一个存放小写单词的词典，并且能快速回答前缀类的问题。实现 `Trie` 类：

- `Trie()`：创建一个空词典
- `insert(word)`：把 `word` 加入词典
- `search(word)`：如果恰好存入过 `word`，返回 `true`，否则返回 `false`（更短或更长的字符串不算）
- `startsWith(prefix)`：如果至少有一个已存入的单词以 `prefix` 开头，返回 `true`，否则返回 `false`

**示例 1：** `insert("grape")`、`insert("gram")`、`search("gram")` → `true`，`search("gra")` → `false`，`startsWith("gra")` → `true`，`startsWith("pear")` → `false`

**示例 2：** `insert("car")`、`search("card")` → `false`，`startsWith("ca")` → `true`，`insert("card")`、`search("card")` → `true`

**约束：**
- $1 \\le \\texttt{word.length},\\ \\texttt{prefix.length} \\le 40$
- `word` 和 `prefix` 只包含小写英文字母
- `insert`、`search` 和 `startsWith` 的总调用次数不超过 $3 \\times 10^{4}$

每次调用的时间应与字符串的长度成正比，而不是与词典中单词的数量成正比。""",
    "hint": "Store the words as a tree of characters: each node has children keyed by letter and a flag that says a word ends there. `search` walks the path and checks the flag; `startsWith` only needs the path to exist.",
    "hint_zh": "把单词存成字符树：每个节点用字母索引子节点，并带一个标记表示此处有单词结束。`search` 沿路径走到底后检查标记，`startsWith` 只需要路径存在。",
    "cases": [
        {
            "ops": ["Trie", "insert", "insert", "search", "search", "startsWith", "startsWith"],
            "args": [[], ["grape"], ["gram"], ["gram"], ["gra"], ["gra"], ["pear"]],
            "expected": [None, None, None, True, False, True, False],
        },
        {
            "ops": ["Trie", "insert", "search", "startsWith", "insert", "search"],
            "args": [[], ["car"], ["card"], ["ca"], ["card"], ["card"]],
            "expected": [None, None, False, True, None, True],
        },
        # A prefix of a stored word is not itself a word
        {
            "ops": ["Trie", "insert", "search", "startsWith", "search"],
            "args": [[], ["tea"], ["te"], ["te"], ["tea"]],
            "expected": [None, None, False, True, True],
        },
        # A word inserted after its own prefix: the end flag must be set on the existing node
        {
            "ops": ["Trie", "insert", "search", "insert", "search", "startsWith"],
            "args": [[], ["dog"], ["do"], ["do"], ["do"], ["do"]],
            "expected": [None, None, False, None, True, True],
        },
        # The word is found as a path but a longer query with the same prefix is not a word
        {
            "ops": ["Trie", "insert", "search", "insert", "search", "search", "startsWith"],
            "args": [[], ["abcd"], ["abc"], ["abc"], ["abc"], ["abcde"], ["abcde"]],
            "expected": [None, None, False, None, True, False, False],
        },
        # Two words diverging at the last letter
        {
            "ops": ["Trie", "insert", "search", "startsWith", "startsWith", "search"],
            "args": [[], ["hello"], ["help"], ["hel"], ["helpx"], ["hello"]],
            "expected": [None, None, False, True, False, True],
        },
        # Queries on an empty dictionary
        {
            "ops": ["Trie", "search", "startsWith"],
            "args": [[], ["a"], ["a"]],
            "expected": [None, False, False],
        },
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False


class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_word = True

    def _find(self, s: str):
        # The node reached by following s from the root, or None if the path is missing
        node = self.root
        for ch in s:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def search(self, word: str) -> bool:
        node = self._find(word)
        return node is not None and node.is_word

    def startsWith(self, prefix: str) -> bool:
        return self._find(prefix) is not None''',
    "explanation": "A trie stores each word as a path of characters from the root. Shared prefixes share nodes, and a flag on the node where a word ends tells `search` apart from `startsWith`. `insert` creates missing nodes along the path. `search` follows the path and returns the flag of the final node. `startsWith` only needs the path to exist.\n\nTime O(L) per operation, where L is the length of the string, space O(total characters inserted) in the worst case.",
    "explanation_zh": "字典树把每个单词存成从根出发的一条字符路径，公共前缀共用节点。节点上的标记表示此处是否有单词结束，`search` 和 `startsWith` 的区别就在这里。`insert` 沿路径创建缺失的节点。`search` 沿路径走到底后返回末节点的标记。`startsWith` 只需要路径存在。\n\n每次操作时间 O(L)，L 为字符串长度；空间最坏为 O(插入的字符总数)。",
}
