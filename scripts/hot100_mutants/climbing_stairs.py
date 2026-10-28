# Mistake: returns the plain Fibonacci number F(n), so n = 2 gives 1 instead of 2
class Solution:
    def climbStairs(self, n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a
