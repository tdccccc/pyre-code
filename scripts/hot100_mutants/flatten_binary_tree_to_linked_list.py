# Mistake: links each node's right pointer in preorder but never clears its left pointer
class Solution:
    def flatten(self, root):
        if not root:
            return
        stack, prev = [root], None
        while stack:
            node = stack.pop()
            if prev:
                prev.right = node
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
            prev = node
