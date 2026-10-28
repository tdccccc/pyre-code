# Mistake: copies each random pointer as-is, so the copy's random pointers lead into the original list
class Solution:
    def copyRandomList(self, head):
        copies = {}
        cur = head
        while cur:
            copies[cur] = Node(cur.val)
            cur = cur.next
        cur = head
        while cur:
            copies[cur].next = copies.get(cur.next)
            copies[cur].random = cur.random
            cur = cur.next
        return copies.get(head)
