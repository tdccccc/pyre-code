# Mistake: reverses each row before transposing, which turns the grid counter-clockwise
class Solution:
    def rotate(self, matrix):
        n = len(matrix)
        for row in matrix:
            row.reverse()
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
