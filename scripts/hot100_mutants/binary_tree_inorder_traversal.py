# Mistake: records each value on the way down (pre-order), before its left subtree
class Solution:
    def inorderTraversal(self, root):
        res = []

        def dfs(node):
            if node:
                res.append(node.val)
                dfs(node.left)
                dfs(node.right)

        dfs(root)
        return res
