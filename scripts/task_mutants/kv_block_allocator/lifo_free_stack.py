# Mistake: free blocks kept on a stack, so the most recently freed block is reused instead of the smallest id
import heapq


class BlockAllocator:
    def __init__(self, num_blocks: int):
        self.refs = [0] * num_blocks  # ref count per physical block, 0 = free
        self.free_heap = list(reversed(range(num_blocks)))  # pop() hands out 0 first

    def allocate(self) -> int:
        if not self.free_heap:
            raise RuntimeError("no free KV blocks")
        block_id = self.free_heap.pop()
        self.refs[block_id] = 1
        return block_id

    def fork(self, block_id: int) -> None:
        self._check_allocated(block_id)
        self.refs[block_id] += 1

    def free(self, block_id: int) -> None:
        self._check_allocated(block_id)
        self.refs[block_id] -= 1
        if self.refs[block_id] == 0:
            self.free_heap.append(block_id)

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
            raise ValueError(f"block {block_id} is free")
