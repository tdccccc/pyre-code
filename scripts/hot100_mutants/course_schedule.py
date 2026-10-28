# Mistake: one visited set for the whole search, so a prerequisite reached twice looks like a cycle
class Solution:
    def canFinish(self, numCourses, prerequisites):
        graph = [[] for _ in range(numCourses)]
        for a, b in prerequisites:
            graph[a].append(b)
        visited = set()

        def dfs(c):
            if c in visited:
                return False
            visited.add(c)
            return all(dfs(p) for p in graph[c])

        return all(dfs(c) for c in range(numCourses))
