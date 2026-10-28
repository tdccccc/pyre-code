# Mistake: marks cells as used but never unmarks them when a branch fails, so dead ends block later paths
class Solution:
    def exist(self, board, word):
        rows, cols = len(board), len(board[0])
        used = set()

        def dfs(r, c, k):
            if board[r][c] != word[k]:
                return False
            if k == len(word) - 1:
                return True
            used.add((r, c))
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in used and dfs(nr, nc, k + 1):
                    return True
            return False

        return any(dfs(r, c, 0) for r in range(rows) for c in range(cols))
