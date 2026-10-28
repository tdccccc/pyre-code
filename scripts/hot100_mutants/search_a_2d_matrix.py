# Mistake: maps the flat index to a cell with the row count instead of the column count, which only works for square grids
class Solution:
    def searchMatrix(self, matrix, target):
        m, n = len(matrix), len(matrix[0])
        lo, hi = 0, m * n - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            r, c = divmod(mid, m)
            if matrix[r][c] == target:
                return True
            if matrix[r][c] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return False
