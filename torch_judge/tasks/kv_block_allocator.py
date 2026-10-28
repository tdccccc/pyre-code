"""KV Cache Block Allocator task."""

TASK = {
    "title": "KV Cache Block Allocator",
    "title_zh": "KV Cache 块分配器",
    "difficulty": "Medium",
    "description_en": "Implement the physical block manager of a paged KV cache (vLLM-style) with reference counting. Blocks can be shared by several sequences (beam search, parallel sampling), and a shared block is copied before one sequence writes into it (copy-on-write).\n\nThe KV cache is split into fixed-size physical blocks with integer ids `0..num_blocks-1`, all free at the start. A block's reference count is the number of sequences holding it; the block is free exactly when its count is 0.\n\n**Signature:** `BlockAllocator(num_blocks: int)` (class) with methods `allocate() -> int`, `fork(block_id: int) -> None`, `free(block_id: int) -> None`, `copy_on_write(block_id: int) -> int`, `ref_count(block_id: int) -> int`, `num_free() -> int`\n\n**Parameters:**\n- `num_blocks` — number of physical blocks\n- `block_id` — a physical block id in `0..num_blocks-1`\n\n**Returns:**\n- `allocate()` — the smallest currently free block id; its ref count becomes 1\n- `fork(block_id)` — `None`; ref count +1 (another sequence now shares the block)\n- `free(block_id)` — `None`; ref count -1; when it reaches 0 the block is free again\n- `copy_on_write(block_id)` — the id the caller writes into (see rules)\n- `ref_count(block_id)` — current ref count (0 for a free block)\n- `num_free()` — number of free blocks\n\n**Constraints:**\n- `allocate()` with no free block raises `RuntimeError`\n- `fork`, `free` and `copy_on_write` on a free block (ref count 0) raise `ValueError`, so a double free is an error\n- `copy_on_write(block_id)`: if the ref count is 1, return `block_id` unchanged (no copy). Otherwise take the smallest free block as the new block, decrement the ref count of `block_id` (it stays allocated for its other holders), and return the new id; the caller copies the data. If no block is free, raise `RuntimeError` and leave all ref counts unchanged\n- Freed blocks return to the free pool, and `allocate()` always returns the smallest id among the free blocks",
    "description_zh": "实现分页 KV 缓存（vLLM 风格）的物理块管理器，并带引用计数。块可以被多个序列共享（beam search、并行采样）；当某个序列要写入共享块时，先复制一份（写时复制，copy-on-write）。\n\nKV 缓存被切分为固定大小的物理块，编号为整数 `0..num_blocks-1`，初始全部空闲。块的引用计数是当前持有它的序列数；引用计数为 0 时该块空闲。\n\n**签名:** `BlockAllocator(num_blocks: int)`（类），方法包括 `allocate() -> int`、`fork(block_id: int) -> None`、`free(block_id: int) -> None`、`copy_on_write(block_id: int) -> int`、`ref_count(block_id: int) -> int`、`num_free() -> int`\n\n**参数:**\n- `num_blocks` — 物理块数量\n- `block_id` — 物理块编号，范围 `0..num_blocks-1`\n\n**返回:**\n- `allocate()` — 当前最小的空闲块编号；该块引用计数变为 1\n- `fork(block_id)` — `None`；引用计数 +1（又一个序列共享该块）\n- `free(block_id)` — `None`；引用计数 -1；减到 0 时该块重新空闲\n- `copy_on_write(block_id)` — 调用方应写入的块编号（见规则）\n- `ref_count(block_id)` — 当前引用计数（空闲块为 0）\n- `num_free()` — 空闲块数量\n\n**约束:**\n- 没有空闲块时，`allocate()` 抛出 `RuntimeError`\n- 对空闲块（引用计数为 0）调用 `fork`、`free`、`copy_on_write` 抛出 `ValueError`，因此重复释放是错误\n- `copy_on_write(block_id)`：引用计数为 1 时原样返回 `block_id`（不复制）。否则取最小的空闲块作为新块，将 `block_id` 的引用计数减 1（它仍被其他持有者占用），返回新块编号；由调用方负责拷贝数据。若没有空闲块，抛出 `RuntimeError`，且所有引用计数保持不变\n- 释放的块回到空闲池，`allocate()` 总是返回当前空闲块中最小的编号",
    "function_name": "BlockAllocator",
    "hint": "free_heap = min-heap of ids 0..n-1;  refs = [0] * n\nallocate(): id = heappop(free_heap); refs[id] = 1; return id\nfork(id) / free(id): require refs[id] > 0; refs[id] +/-= 1\n  free(id): if refs[id] == 0 after decrement -> heappush(free_heap, id)\ncopy_on_write(id):\n  if refs[id] == 1: return id\n  new = allocate()          # may raise, before any change\n  refs[id] -= 1; return new",
    "hint_zh": "free_heap = 编号 0..n-1 的最小堆;  refs = [0] * n\nallocate(): id = heappop(free_heap); refs[id] = 1; return id\nfork(id) / free(id): 要求 refs[id] > 0; refs[id] +/-= 1\n  free(id): 减后若为 0 -> heappush(free_heap, id)\ncopy_on_write(id):\n  if refs[id] == 1: return id\n  new = allocate()          # 可能抛异常，且此前未做任何修改\n  refs[id] -= 1; return new",
    "tests": [
        {
            "name": "Allocation order and ref counts",
            "code": """
alloc = {fn}(4)
ids = [alloc.allocate() for _ in range(4)]
assert ids == [0, 1, 2, 3], f'allocate() should hand out the smallest free ids in order, got {ids}'
assert all(alloc.ref_count(i) == 1 for i in range(4)), 'newly allocated blocks must have ref count 1'
assert alloc.num_free() == 0, f'all blocks are in use, but num_free() = {alloc.num_free()}'
""",
        },
        {
            "name": "Exhaustion raises RuntimeError",
            "code": """
alloc = {fn}(2)
alloc.allocate()
alloc.allocate()
try:
    alloc.allocate()
except RuntimeError:
    pass
else:
    raise AssertionError('allocate() with no free block must raise RuntimeError')
assert alloc.num_free() == 0, 'failed allocate() must not change the free count'
""",
        },
        {
            "name": "Free and reuse picks the smallest free id",
            "code": """
alloc = {fn}(4)
for _ in range(4):
    alloc.allocate()
alloc.free(0)
alloc.free(2)
assert alloc.num_free() == 2, f'num_free() should be 2, got {alloc.num_free()}'
got = alloc.allocate()
assert got == 0, f'the smallest free block is 0 even though 2 was freed last, got {got}'
got = alloc.allocate()
assert got == 2, f'the next free block is 2, got {got}'
""",
        },
        {
            "name": "Fork and free counting",
            "code": """
alloc = {fn}(3)
b = alloc.allocate()
alloc.fork(b)
alloc.fork(b)
assert alloc.ref_count(b) == 3, f'ref count after two forks should be 3, got {alloc.ref_count(b)}'
alloc.free(b)
assert alloc.ref_count(b) == 2, f'ref count should be 2, got {alloc.ref_count(b)}'
alloc.free(b)
assert alloc.ref_count(b) == 1, f'ref count should be 1, got {alloc.ref_count(b)}'
assert alloc.num_free() == 2, f'a held block is not free, num_free() should be 2, got {alloc.num_free()}'
alloc.free(b)
assert alloc.ref_count(b) == 0, f'ref count should be 0 after the last free, got {alloc.ref_count(b)}'
assert alloc.num_free() == 3, f'all blocks should be free again, num_free() = {alloc.num_free()}'
""",
        },
        {
            "name": "Copy-on-write with ref count 1 returns the same block",
            "code": """
alloc = {fn}(4)
alloc.allocate()          # block 0
alloc.allocate()          # block 1
c = alloc.allocate()      # block 2
alloc.free(0)             # block 0 is free again, so it is smaller than c
got = alloc.copy_on_write(c)
assert got == c, f'a sole owner must write in place: expected {c}, got {got}'
assert alloc.ref_count(c) == 1, f'ref count must stay 1, got {alloc.ref_count(c)}'
assert alloc.num_free() == 2, f'no block should be allocated, num_free() = {alloc.num_free()}'
""",
        },
        {
            "name": "Copy-on-write on a shared block copies to the smallest free block",
            "code": """
alloc = {fn}(3)
b = alloc.allocate()      # block 0
alloc.fork(b)             # a second sequence shares it
new = alloc.copy_on_write(b)
assert new == 1, f'the copy should go to the smallest free block 1, got {new}'
assert alloc.ref_count(b) == 1, f'old block should drop to ref count 1, got {alloc.ref_count(b)}'
assert alloc.ref_count(new) == 1, f'new block should have ref count 1, got {alloc.ref_count(new)}'
assert alloc.copy_on_write(b) == b, 'the remaining holder now owns the block exclusively'

small = {fn}(1)
x = small.allocate()
small.fork(x)
try:
    small.copy_on_write(x)
except RuntimeError:
    pass
else:
    raise AssertionError('copy_on_write of a shared block needs a free block')
assert small.ref_count(x) == 2, f'failed copy-on-write must leave ref counts unchanged, got {small.ref_count(x)}'
""",
        },
        {
            "name": "Operations on a free block raise ValueError",
            "code": """
alloc = {fn}(2)
b = alloc.allocate()
alloc.free(b)
for name, op in (('free', alloc.free), ('fork', alloc.fork), ('copy_on_write', alloc.copy_on_write)):
    try:
        op(b)
    except ValueError:
        continue
    raise AssertionError(f'{name}() on a free block must raise ValueError')
assert alloc.num_free() == 2, f'failed operations must not change the pool, num_free() = {alloc.num_free()}'
assert alloc.ref_count(b) == 0, f'block should stay free, ref count = {alloc.ref_count(b)}'
""",
        },
        {
            "name": "Randomized simulation against a dict model",
            "code": """
import random
rng = random.Random(1234)
n = 6
alloc = {fn}(n)
refs = {}  # reference model: allocated block id -> ref count

def smallest_free():
    for i in range(n):
        if i not in refs:
            return i
    return None

def check(step):
    for i in range(n):
        assert alloc.ref_count(i) == refs.get(i, 0), f'step {step}: ref_count({i}) = {alloc.ref_count(i)}, expected {refs.get(i, 0)}'
    assert alloc.num_free() == n - len(refs), f'step {step}: num_free() = {alloc.num_free()}, expected {n - len(refs)}'

for step in range(800):
    live = sorted(refs)
    r = rng.random()
    if not live or r < 0.35:
        expected = smallest_free()
        if expected is None:
            try:
                alloc.allocate()
            except RuntimeError:
                pass
            else:
                raise AssertionError(f'step {step}: allocate() should raise RuntimeError when full')
        else:
            got = alloc.allocate()
            assert got == expected, f'step {step}: allocate() returned {got}, expected {expected}'
            refs[got] = 1
    elif r < 0.55:
        b = rng.choice(live)
        alloc.fork(b)
        refs[b] += 1
    elif r < 0.8:
        b = rng.choice(live)
        alloc.free(b)
        refs[b] -= 1
        if refs[b] == 0:
            del refs[b]
    else:
        b = rng.choice(live)
        if refs[b] == 1:
            got = alloc.copy_on_write(b)
            assert got == b, f'step {step}: sole owner of {b} got {got}'
        elif smallest_free() is not None:
            expected = smallest_free()
            got = alloc.copy_on_write(b)
            assert got == expected, f'step {step}: copy_on_write({b}) returned {got}, expected {expected}'
            refs[b] -= 1
            refs[got] = 1
    check(step)
""",
        },
    ],
    "solution": '''import heapq


class BlockAllocator:
    def __init__(self, num_blocks: int):
        self.refs = [0] * num_blocks  # ref count per physical block, 0 = free
        self.free_heap = list(range(num_blocks))  # ids are already a valid min-heap

    def allocate(self) -> int:
        if not self.free_heap:
            raise RuntimeError("no free KV blocks")
        block_id = heapq.heappop(self.free_heap)  # smallest free id
        self.refs[block_id] = 1
        return block_id

    def fork(self, block_id: int) -> None:
        self._check_allocated(block_id)
        self.refs[block_id] += 1

    def free(self, block_id: int) -> None:
        self._check_allocated(block_id)
        self.refs[block_id] -= 1
        if self.refs[block_id] == 0:
            heapq.heappush(self.free_heap, block_id)

    def copy_on_write(self, block_id: int) -> int:
        self._check_allocated(block_id)
        if self.refs[block_id] == 1:
            return block_id  # sole owner writes in place
        new_id = self.allocate()  # raises before any state changes if no block is free
        self.refs[block_id] -= 1  # other holders keep the old block
        return new_id

    def ref_count(self, block_id: int) -> int:
        return self.refs[block_id]

    def num_free(self) -> int:
        return len(self.free_heap)

    def _check_allocated(self, block_id: int) -> None:
        if self.refs[block_id] == 0:
            raise ValueError(f"block {block_id} is free")''',
    "demo": """alloc = BlockAllocator(4)
a = alloc.allocate()  # block 0
b = alloc.allocate()  # block 1
alloc.fork(a)  # beam search: a second sequence shares block 0
print('ref_count(0) =', alloc.ref_count(a))

new = alloc.copy_on_write(a)  # the second sequence is about to write
print('copy-on-write gives block', new, '| ref_count(0) =', alloc.ref_count(a))

alloc.free(b)
print('free blocks:', alloc.num_free(), '| next allocate ->', alloc.allocate())""",
}
