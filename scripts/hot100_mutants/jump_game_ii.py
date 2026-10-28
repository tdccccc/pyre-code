# Mistake: moves the window end before folding in index i's reach, so the jump count lags a step
class Solution:
    def jump(self, nums):
        jumps = end = farthest = 0
        for i in range(len(nums) - 1):
            if i == end:
                jumps += 1
                end = farthest
            farthest = max(farthest, i + nums[i])
        return jumps
