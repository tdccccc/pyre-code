"""Prefix Caching task."""

TASK = {
    "title": "Prefix Caching",
    "title_zh": "前缀缓存",
    "difficulty": "Hard",
    "description_en": "Implement automatic prefix caching (vLLM / SGLang style). Sequences that start with the same tokens reuse the same physical KV blocks, so a shared prompt prefix is computed only once.\n\nThe cache manages `num_blocks` physical blocks with ids `0..num_blocks-1`, each holding `block_size` tokens of KV. Only *full* blocks can be cached; a partial block is never shared.\n\n**Signature:** `PrefixCache(num_blocks: int, block_size: int)` (class) with methods `allocate_sequence(tokens: list[int]) -> tuple[list[int], int]`, `free_sequence(block_table: list[int]) -> None`, `ref_count(block_id: int) -> int`, `num_cached_blocks() -> int`\n\n**Parameters:**\n- `tokens` — prompt token ids of a new sequence\n- `block_table` — the list returned as the first element of an earlier `allocate_sequence` call\n\n**Returns:**\n- `allocate_sequence(tokens)` — `(block_table, num_cached_tokens)`. `block_table[i]` is the physical block of logical block `i`; `num_cached_tokens` is the number of prompt tokens whose KV was reused\n- `free_sequence(block_table)` — `None`\n- `ref_count(block_id)` — number of sequences holding the block (0 if none)\n- `num_cached_blocks()` — number of blocks registered in the cache, whatever their ref count\n\n**Rules:**\n- Logical block `i` holds `tokens[i*block_size:(i+1)*block_size]`. There are `ceil(len(tokens)/block_size)` logical blocks. A block is *full* when it holds `block_size` tokens. An empty token list gives `([], 0)`.\n- Full block `i` has the key `tuple(tokens[:(i+1)*block_size])`: the whole prefix up to the end of that block. The same content after a different prefix therefore never matches.\n- Walk the logical blocks from the start. While the block is full and its key is registered in the cache, reuse the registered physical block: its ref count increases by 1, and if it was 0 it is revived and no longer evictable. At the first block that is not reused, stop reusing. That block and every later block are newly allocated, even if a later key happens to be registered.\n- Each newly allocated full block is registered under its key, unless that key is already registered to another block; then the new block stays unregistered. The final partial block is newly allocated and never registered.\n- `num_cached_tokens = (number of reused blocks) * block_size`.\n- `free_sequence(block_table)` processes the entries in order and decrements each ref count. A block whose ref count reaches 0 and is registered becomes *evictable* but stays cached. An unregistered block whose ref count reaches 0 returns to the free pool at once. Each block records when it became evictable, so the blocks released by one call are ordered by their position in `block_table`.\n- Allocating a new block: if the free pool (never-used or unregistered blocks) is non-empty, take its smallest id. Otherwise evict the evictable block that became evictable earliest (LRU): remove its key from the cache and use the block. If neither exists, raise `RuntimeError`. The state after a failed `allocate_sequence` is unspecified.",
    "description_zh": "实现自动前缀缓存（vLLM / SGLang 风格）。以相同 token 开头的序列复用同一批物理 KV 块，因此共享的提示前缀只需计算一次。\n\n缓存管理 `num_blocks` 个物理块，编号为 `0..num_blocks-1`，每块保存 `block_size` 个 token 的 KV。只有*满块*可以被缓存；不满的块永远不共享。\n\n**签名:** `PrefixCache(num_blocks: int, block_size: int)`（类），方法包括 `allocate_sequence(tokens: list[int]) -> tuple[list[int], int]`、`free_sequence(block_table: list[int]) -> None`、`ref_count(block_id: int) -> int`、`num_cached_blocks() -> int`\n\n**参数:**\n- `tokens` — 新序列的提示 token 编号列表\n- `block_table` — 之前 `allocate_sequence` 调用返回的第一个元素\n\n**返回:**\n- `allocate_sequence(tokens)` — `(block_table, num_cached_tokens)`。`block_table[i]` 是逻辑块 `i` 对应的物理块；`num_cached_tokens` 是复用了 KV 的提示 token 数\n- `free_sequence(block_table)` — `None`\n- `ref_count(block_id)` — 持有该块的序列数（无则为 0）\n- `num_cached_blocks()` — 已登记在缓存中的块数（不论引用计数）\n\n**规则:**\n- 逻辑块 `i` 保存 `tokens[i*block_size:(i+1)*block_size]`，共 `ceil(len(tokens)/block_size)` 个逻辑块。块内 token 数达到 `block_size` 时称为*满块*。空 token 列表返回 `([], 0)`。\n- 满块 `i` 的键为 `tuple(tokens[:(i+1)*block_size])`，即直到该块末尾的整个前缀。因此内容相同但前缀不同的块永远不会匹配。\n- 从第一个逻辑块开始依次遍历。当块是满块且其键已登记在缓存中时，复用该已登记的物理块：引用计数 +1；若原来为 0，则它被复活，不再可驱逐。遇到第一个不能复用的块后停止复用：该块及之后所有块都重新分配，即使之后某个键恰好已登记也不复用。\n- 新分配的满块以其键登记，除非该键已登记给另一个块；此时新块保持未登记。最后的不满块是新分配的，且永远不登记。\n- `num_cached_tokens = (复用的块数) * block_size`。\n- `free_sequence(block_table)` 按顺序处理各项并把每块的引用计数减 1。引用计数降为 0 且已登记的块变为*可驱逐*，但仍留在缓存中；未登记的块引用计数降为 0 时立即回到空闲池。每个块记录它变为可驱逐的时刻，因此同一次调用释放的块按其在 `block_table` 中的位置排序。\n- 分配新块时：若空闲池（从未使用或未登记的块）非空，取其中最小的编号。否则驱逐最早变为可驱逐的块（LRU）：从缓存中删除其键，并使用该块。若两者都没有，抛出 `RuntimeError`。`allocate_sequence` 失败后的状态未作规定。",
    "function_name": "PrefixCache",
    "hint": "free_heap = min-heap of unused ids;  refs[b];  key_of[b] (block -> key);  cache[key] -> b;  evictable[b] = tick\nallocate_sequence(tokens):\n  reusing = True\n  for each logical block i (end = min((i+1)*bs, len)):\n    if reusing and full and key in cache:\n      b = cache[key]; if refs[b] == 0: drop from evictable\n      refs[b] += 1; append b; continue\n    reusing = False\n    b = take(); refs[b] = 1\n    if full and key not in cache: register b under key\n    append b\n  return table, reused * bs\ntake(): free_heap pop -> else evict min-tick evictable (delete its key) -> else RuntimeError\nfree_sequence(table): for b in table: refs[b] -= 1; if 0: registered ? evictable[b] = tick++ : heappush(free_heap, b)",
    "hint_zh": "free_heap = 未使用编号的最小堆;  refs[b];  key_of[b]（块 -> 键）;  cache[键] -> b;  evictable[b] = 时间戳\nallocate_sequence(tokens):\n  reusing = True\n  对每个逻辑块 i（end = min((i+1)*bs, len)）:\n    if reusing and 满块 and key in cache:\n      b = cache[key]; 若 refs[b] == 0 则移出 evictable\n      refs[b] += 1; 追加 b; continue\n    reusing = False\n    b = take(); refs[b] = 1\n    若 满块 且 key 不在 cache 中: 以 key 登记 b\n    追加 b\n  return table, reused * bs\ntake(): free_heap 弹出 -> 否则驱逐时间戳最小的 evictable（删除其键） -> 否则 RuntimeError\nfree_sequence(table): 对 table 中每个 b: refs[b] -= 1; 若为 0: 已登记 ? evictable[b] = tick++ : heappush(free_heap, b)",
    "tests": [
        {
            "name": "Disjoint prompts share no blocks",
            "code": """
cache = {fn}(8, 2)
t1, n1 = cache.allocate_sequence([1, 2, 3, 4])
t2, n2 = cache.allocate_sequence([5, 6, 7, 8])
assert n1 == 0 and n2 == 0, f'no cached tokens expected for new prompts, got {n1} and {n2}'
assert len(t1) == 2 and len(t2) == 2, f'two logical blocks each, got {t1} and {t2}'
assert not set(t1) & set(t2), f'disjoint prompts must not share blocks: {t1} vs {t2}'
assert all(cache.ref_count(b) == 1 for b in t1 + t2), 'every block should be held once'
assert cache.num_cached_blocks() == 4, f'four full blocks should be registered, got {cache.num_cached_blocks()}'
""",
        },
        {
            "name": "Shared prefix reuses blocks and reports cached tokens",
            "code": """
cache = {fn}(8, 2)
t1, n1 = cache.allocate_sequence([1, 2, 3, 4, 5])      # [1,2] [3,4] [5]
assert n1 == 0 and len(t1) == 3, f'first prompt: cached={n1}, table={t1}'
t2, n2 = cache.allocate_sequence([1, 2, 3, 4, 9, 9])   # [1,2] [3,4] [9,9]
assert n2 == 4, f'two full blocks are shared, so 4 tokens are cached, got {n2}'
assert t2[:2] == t1[:2], f'shared prefix must reuse blocks {t1[:2]}, got {t2}'
assert t2[2] not in t1, f'the new block must be a fresh block, got {t2[2]}'
assert cache.ref_count(t1[0]) == 2 and cache.ref_count(t1[1]) == 2, 'shared blocks should have ref count 2'
assert cache.ref_count(t1[2]) == 1 and cache.ref_count(t2[2]) == 1, 'unshared blocks should have ref count 1'
assert cache.num_cached_blocks() == 3, f'[1,2], [3,4] and [1,2,3,4,9,9] are registered, got {cache.num_cached_blocks()}'
""",
        },
        {
            "name": "Partial last block is never shared",
            "code": """
cache = {fn}(8, 4)
t1, n1 = cache.allocate_sequence([1, 2, 3, 4, 5, 6])   # full [1..4] + partial [5, 6]
t2, n2 = cache.allocate_sequence([1, 2, 3, 4, 5, 6])
assert n2 == 4, f'only the full block can be reused, so 4 tokens are cached, got {n2}'
assert t2[0] == t1[0], f'the full block should be shared, got {t2} vs {t1}'
assert t2[1] != t1[1], f'the partial block must not be shared, got {t2[1]}'
assert cache.num_cached_blocks() == 1, f'only the full block is registered, got {cache.num_cached_blocks()}'
""",
        },
        {
            "name": "Same content after a different prefix does not match",
            "code": """
cache = {fn}(8, 2)
t1, _ = cache.allocate_sequence([1, 2, 3, 4])
t2, n2 = cache.allocate_sequence([3, 4])
assert n2 == 0, f'block [3, 4] has a different prefix from [1, 2, 3, 4], got {n2} cached tokens'
assert t2[0] not in t1, f'a fresh block is required, got {t2[0]} from {t1}'
t3, n3 = cache.allocate_sequence([9, 9, 3, 4])
assert n3 == 0 and not set(t3) & set(t1), f'nothing in [9, 9, 3, 4] is cached, got {t3} with {n3}'
""",
        },
        {
            "name": "Reuse after free revives cached blocks",
            "code": """
cache = {fn}(4, 2)
t1, _ = cache.allocate_sequence([1, 2, 3, 4])   # blocks 0 and 1
cache.free_sequence(t1)
assert cache.ref_count(t1[0]) == 0 and cache.ref_count(t1[1]) == 0, 'freed blocks should have ref count 0'
assert cache.num_cached_blocks() == 2, f'freed registered blocks stay cached, got {cache.num_cached_blocks()}'
t2, n2 = cache.allocate_sequence([1, 2, 3, 4, 5])
assert t2[:2] == t1, f'expected the revived blocks {t1}, got {t2}'
assert n2 == 4, f'4 tokens should be reused, got {n2}'
assert cache.ref_count(t1[0]) == 1 and cache.ref_count(t1[1]) == 1, 'revived blocks should be held once'
""",
        },
        {
            "name": "Unregistered blocks return to the free pool",
            "code": """
cache = {fn}(4, 2)
t1, _ = cache.allocate_sequence([1, 2, 3])   # block 0 full and registered, block 1 partial
assert t1 == [0, 1], f'unexpected blocks {t1}'
cache.free_sequence(t1)
assert cache.num_cached_blocks() == 1, f'only the full block stays registered, got {cache.num_cached_blocks()}'
t2, n2 = cache.allocate_sequence([7])        # free pool is {1, 2, 3}; smallest is 1
assert t2 == [1] and n2 == 0, f'the unregistered block should be reused first, got {t2} with {n2}'
t3, n3 = cache.allocate_sequence([8, 8, 9, 9])   # free pool is {2, 3}
assert t3 == [2, 3] and n3 == 0, f'expected blocks [2, 3], got {t3} with {n3}'
t4, n4 = cache.allocate_sequence([5])        # pool is empty: evict block 0
assert t4 == [0] and n4 == 0, f'the only evictable block is 0, got {t4} with {n4}'
assert cache.num_cached_blocks() == 2, f'[8, 8] and [8, 8, 9, 9] remain, got {cache.num_cached_blocks()}'
""",
        },
        {
            "name": "LRU eviction order",
            "code": """
cache = {fn}(3, 2)
a, _ = cache.allocate_sequence([1, 2])   # block 0
b, _ = cache.allocate_sequence([3, 4])   # block 1
c, _ = cache.allocate_sequence([5, 6])   # block 2
assert (a, b, c) == ([0], [1], [2]), f'unexpected blocks {a} {b} {c}'
cache.free_sequence(b)                   # block 1 becomes evictable first
cache.free_sequence(a)                   # then block 0
t, n = cache.allocate_sequence([7, 8])   # pool is empty: evict the earliest, block 1
assert t == [1] and n == 0, f'least recently released block 1 must be evicted first, got {t}'
assert cache.num_cached_blocks() == 3, f'[1, 2], [5, 6] and [7, 8] are cached, got {cache.num_cached_blocks()}'
t2, n2 = cache.allocate_sequence([1, 2])
assert t2 == [0] and n2 == 2, f'[1, 2] should still be cached in block 0, got {t2} with {n2}'

cache2 = {fn}(2, 2)
t3, _ = cache2.allocate_sequence([1, 2, 3, 4])   # blocks [0, 1]
cache2.free_sequence(t3)                         # released in table order
t4, n4 = cache2.allocate_sequence([9, 9])
assert t4 == [0] and n4 == 0, f'the earlier table position must be evicted first, got {t4}'
""",
        },
        {
            "name": "RuntimeError when every block is in use",
            "code": """
cache = {fn}(2, 2)
t1, n1 = cache.allocate_sequence([1, 2, 3, 4])   # both blocks in use
assert n1 == 0, f'nothing cached yet, got {n1}'
t2, n2 = cache.allocate_sequence([1, 2, 3, 4])   # fully cached: no new block needed
assert t2 == t1 and n2 == 4, f'expected a full cache hit on {t1}, got {t2} with {n2}'
assert cache.ref_count(t1[0]) == 2, f'the block should be held twice, got {cache.ref_count(t1[0])}'
try:
    cache.allocate_sequence([9, 9])
except RuntimeError:
    pass
else:
    raise AssertionError('allocate_sequence must raise RuntimeError when no block is free or evictable')
""",
        },
        {
            "name": "Randomized simulation against a reference model",
            "code": """
import copy
import random

class Model:
    def __init__(self, num_blocks, bs):
        self.bs = bs
        self.refs = [0] * num_blocks
        self.keys = [None] * num_blocks  # key registered to each block
        self.cache = {}                  # key -> block
        self.lru = []                    # evictable blocks, oldest first

    def take(self):
        unused = [b for b in range(len(self.refs)) if self.refs[b] == 0 and self.keys[b] is None]
        if unused:
            return min(unused)
        if self.lru:
            victim = self.lru.pop(0)
            del self.cache[self.keys[victim]]
            self.keys[victim] = None
            return victim
        raise RuntimeError('full')

    def allocate(self, tokens):
        n = len(tokens)
        table, hits, reusing = [], 0, True
        for i in range((n + self.bs - 1) // self.bs):
            end = min(n, (i + 1) * self.bs)
            full = end == (i + 1) * self.bs
            key = tuple(tokens[:end])
            if reusing and full and key in self.cache:
                b = self.cache[key]
                if self.refs[b] == 0:
                    self.lru.remove(b)
                self.refs[b] += 1
                hits += 1
            else:
                reusing = False
                b = self.take()
                self.refs[b] = 1
                if full and key not in self.cache:
                    self.cache[key] = b
                    self.keys[b] = key
            table.append(b)
        return table, hits * self.bs

    def free(self, table):
        for b in table:
            self.refs[b] -= 1
            if self.refs[b] == 0 and self.keys[b] is not None:
                self.lru.append(b)

rng = random.Random(99)
num_blocks, bs = 6, 2
impl = {fn}(num_blocks, bs)
model = Model(num_blocks, bs)
live = []  # block tables of sequences still in use
for step in range(500):
    if not live or rng.random() < 0.6:
        tokens = [rng.randrange(2) for _ in range(rng.randint(1, 6))]
        snapshot = copy.deepcopy(model)
        try:
            expected = model.allocate(tokens)
        except RuntimeError:
            model = snapshot  # both sides skip this op
            continue
        got = impl.allocate_sequence(tokens)
        assert got == expected, f'step {step}: allocate_sequence({tokens}) returned {got}, expected {expected}'
        live.append(expected[0])
    else:
        table = live.pop(rng.randrange(len(live)))
        model.free(table)
        impl.free_sequence(table)
    for b in range(num_blocks):
        assert impl.ref_count(b) == model.refs[b], f'step {step}: ref_count({b}) = {impl.ref_count(b)}, expected {model.refs[b]}'
    assert impl.num_cached_blocks() == sum(k is not None for k in model.keys), f'step {step}: wrong num_cached_blocks()'
""",
        },
    ],
    "solution": '''import heapq


class PrefixCache:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.refs = [0] * num_blocks
        self.free_heap = list(range(num_blocks))  # never-used or unregistered blocks
        self.key_of = {}  # registered block id -> prefix key
        self.cache = {}  # prefix key -> registered block id
        self.evictable = {}  # ref-0 registered block id -> tick when it became evictable
        self.tick = 0

    def allocate_sequence(self, tokens: list[int]) -> tuple[list[int], int]:
        bs = self.block_size
        block_table = []
        reused = 0
        reusing = True  # False once the first block misses
        for i in range(-(-len(tokens) // bs)):  # ceil division
            end = min((i + 1) * bs, len(tokens))
            full = end - i * bs == bs
            key = tuple(tokens[:end])
            if reusing and full and key in self.cache:
                block_id = self.cache[key]
                if self.refs[block_id] == 0:
                    del self.evictable[block_id]  # revived: no longer evictable
                self.refs[block_id] += 1
                block_table.append(block_id)
                reused += 1
                continue
            reusing = False
            block_id = self._take_block()
            self.refs[block_id] = 1
            if full and key not in self.cache:
                self.cache[key] = block_id
                self.key_of[block_id] = key
            block_table.append(block_id)
        return block_table, reused * bs

    def free_sequence(self, block_table: list[int]) -> None:
        for block_id in block_table:  # earlier positions become evictable first
            self.refs[block_id] -= 1
            if self.refs[block_id] == 0:
                if block_id in self.key_of:
                    self.evictable[block_id] = self.tick
                    self.tick += 1
                else:
                    heapq.heappush(self.free_heap, block_id)

    def ref_count(self, block_id: int) -> int:
        return self.refs[block_id]

    def num_cached_blocks(self) -> int:
        return len(self.key_of)

    def _take_block(self) -> int:
        if self.free_heap:
            return heapq.heappop(self.free_heap)
        if self.evictable:
            victim = min(self.evictable, key=self.evictable.get)  # least recently released
            del self.evictable[victim]
            del self.cache[self.key_of.pop(victim)]
            return victim
        raise RuntimeError("no free or evictable KV blocks")''',
    "demo": """cache = PrefixCache(6, 2)
system = [101, 102, 103, 104]  # shared system prompt: two full blocks

t1, n1 = cache.allocate_sequence(system + [7])
print('seq1 blocks:', t1, '| cached tokens:', n1)
t2, n2 = cache.allocate_sequence(system + [8, 9])
print('seq2 blocks:', t2, '| cached tokens:', n2)
print('ref counts of shared blocks:', [cache.ref_count(b) for b in t1[:2]])

cache.free_sequence(t1)
cache.free_sequence(t2)
print('cached blocks after both finish:', cache.num_cached_blocks())""",
}
