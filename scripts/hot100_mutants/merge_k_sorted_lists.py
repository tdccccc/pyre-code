# Mistake: heap holds (value, node) pairs, so equal values make heapq compare ListNode objects and crash
import heapq


class Solution:
    def mergeKLists(self, lists):
        heap = [(node.val, node) for node in lists if node]
        heapq.heapify(heap)
        dummy = tail = ListNode()
        while heap:
            _, node = heapq.heappop(heap)
            tail.next = node
            tail = node
            if node.next:
                heapq.heappush(heap, (node.next.val, node.next))
        return dummy.next
