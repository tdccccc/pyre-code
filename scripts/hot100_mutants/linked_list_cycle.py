# Mistake: remembers visited values instead of visited nodes, so two equal values look like a cycle
class Solution:
    def hasCycle(self, head):
        seen = set()
        while head:
            if head.val in seen:
                return True
            seen.add(head.val)
            head = head.next
        return False
