# Mistake: never checks that an index is reachable, so jumps from stuck positions still count
class Solution:
    def canJump(self, nums):
        far = 0
        for i, x in enumerate(nums):
            far = max(far, i + x)
        return far >= len(nums) - 1
