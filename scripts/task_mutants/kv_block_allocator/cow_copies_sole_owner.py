# Mistake: copy_on_write always allocates a new block, even when the caller is the only holder
import heapq


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
        self.free(block_id)  # release the old block
        return self.allocate()  # always copy

    def ref_count(self, block_id: int) -> int:
        return self.refs[block_id]

    def num_free(self) -> int:
        return len(self.free_heap)

    def _check_allocated(self, block_id: int) -> None:
        if self.refs[block_id] == 0:
            raise ValueError(f"block {block_id} is free")
