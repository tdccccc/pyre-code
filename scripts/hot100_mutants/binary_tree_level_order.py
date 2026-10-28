# Mistake: Enqueues the right child before the left one
from collections import deque
class Solution:
    def levelOrder(self, root):
        if not root:
            return []
        res, q = [], deque([root])
        while q:
            lvl = []
            for _ in range(len(q)):
                n = q.popleft()
                lvl.append(n.val)
                if n.right: q.append(n.right)
                if n.left: q.append(n.left)
            res.append(lvl)
        return res
