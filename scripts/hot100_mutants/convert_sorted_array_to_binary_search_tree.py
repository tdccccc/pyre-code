# Mistake: always takes the first element as the root, so the tree leans right into a chain
class Solution:
    def sortedArrayToBST(self, nums):
        if not nums:
            return None
        return TreeNode(nums[0], None, self.sortedArrayToBST(nums[1:]))
