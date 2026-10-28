# Mistake: robs either all even-indexed houses or all odd-indexed houses
class Solution:
    def rob(self, nums):
        return max(sum(nums[0::2]), sum(nums[1::2]))
