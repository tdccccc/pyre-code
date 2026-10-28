# Mistake: treats the cell above or to the left of the grid as 0, so the edges get a free ride
class Solution:
    def minPathSum(self, grid):
        m, n = len(grid), len(grid[0])
        cost = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                up = cost[i - 1][j] if i > 0 else 0
                left = cost[i][j - 1] if j > 0 else 0
                cost[i][j] = grid[i][j] + min(up, left)
        return cost[-1][-1]
