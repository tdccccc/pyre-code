# Mistake: the flood fill only moves right and down, so land reached by going left or up is counted again
class Solution:
    def numIslands(self, grid):
        m, n = len(grid), len(grid[0])
        seen = [[False] * n for _ in range(m)]

        def fill(i, j):
            if i >= m or j >= n or grid[i][j] != "1" or seen[i][j]:
                return
            seen[i][j] = True
            fill(i + 1, j)
            fill(i, j + 1)

        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1" and not seen[i][j]:
                    count += 1
                    fill(i, j)
        return count
