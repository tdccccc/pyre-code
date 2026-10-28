# Mistake: fills the knapsack forwards, so one number can be used several times
class Solution:
    def canPartition(self, nums):
        total = sum(nums)
        if total % 2:
            return False
        target = total // 2
        dp = [True] + [False] * target
        for x in nums:
            for s in range(x, target + 1):
                dp[s] = dp[s] or dp[s - x]
        return dp[target]
