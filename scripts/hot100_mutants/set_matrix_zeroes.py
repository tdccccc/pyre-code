# Mistake: clears row and column as it scans, so the zeros it creates clear the rest of the matrix
class Solution:
    def setZeroes(self, matrix):
        m, n = len(matrix), len(matrix[0])
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    for k in range(n):
                        matrix[i][k] = 0
                    for k in range(m):
                        matrix[k][j] = 0
