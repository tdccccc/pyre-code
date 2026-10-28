# Mistake: eviction removes the most recently released block (LIFO) instead of the least recently used one
import heapq


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
            victim = max(self.evictable, key=self.evictable.get)  # least recently released
            del self.evictable[victim]
            del self.cache[self.key_of.pop(victim)]
            return victim
        raise RuntimeError("no free or evictable KV blocks")
