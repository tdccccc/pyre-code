# Mistake: sweeps the grid in place, so an orange rotted earlier in a sweep spreads within the same minute
class Solution:
    def orangesRotting(self, grid):
        m, n = len(grid), len(grid[0])
        minutes = 0
        changed = True
        while changed:
            changed = False
            for i in range(m):
                for j in range(n):
                    if grid[i][j] == 2:
                        for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                            if 0 <= x < m and 0 <= y < n and grid[x][y] == 1:
                                grid[x][y] = 2
                                changed = True
            if changed:
                minutes += 1
        if any(1 in row for row in grid):
            return -1
        return minutes
