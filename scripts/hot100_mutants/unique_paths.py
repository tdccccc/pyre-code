# Mistake: updates each row right to left, so the left neighbour still holds the previous row's value
class Solution:
    def uniquePaths(self, m, n):
        ways = [1] * n
        for _ in range(1, m):
            for j in range(n - 1, 0, -1):
                ways[j] += ways[j - 1]
        return ways[-1]
