# Mistake: takes the highest price minus the lowest, ignoring whether the low comes before the high
class Solution:
    def maxProfit(self, prices):
        return max(prices) - min(prices)
