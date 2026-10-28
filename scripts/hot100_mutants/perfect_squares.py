# Mistake: greedy, always subtracts the largest square that still fits
class Solution:
    def numSquares(self, n):
        count = 0
        while n > 0:
            k = 1
            while (k + 1) * (k + 1) <= n:
                k += 1
            n -= k * k
            count += 1
        return count
