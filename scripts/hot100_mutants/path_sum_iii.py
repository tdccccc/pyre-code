# Mistake: never removes a node's prefix sum on the way back up, so sibling subtrees see it
class Solution:
    def pathSum(self, root, targetSum):
        counts = {0: 1}

        def dfs(node, prefix):
            if not node:
                return 0
            prefix += node.val
            found = counts.get(prefix - targetSum, 0)
            counts[prefix] = counts.get(prefix, 0) + 1
            return found + dfs(node.left, prefix) + dfs(node.right, prefix)

        return dfs(root, 0)
