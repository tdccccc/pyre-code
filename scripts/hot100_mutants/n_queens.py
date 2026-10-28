# Mistake: checks the column and one diagonal only, so queens can attack each other along the other diagonal
class Solution:
    def solveNQueens(self, n):
        result, cols = [], []

        def place(row):
            if row == n:
                result.append(["." * c + "Q" + "." * (n - c - 1) for c in cols])
                return
            for c in range(n):
                if c in cols or any(row - c == r - cc for r, cc in enumerate(cols)):
                    continue
                cols.append(c)
                place(row + 1)
                cols.pop()

        place(0)
        return result
