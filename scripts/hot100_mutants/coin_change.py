# Mistake: Greedy: always takes the largest coin that fits
class Solution:
    def coinChange(self, coins, amount):
        n = 0
        for c in sorted(coins, reverse=True):
            n += amount // c
            amount %= c
        return n if amount == 0 else -1
