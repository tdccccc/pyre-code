# Mistake: pops on <= so an equal temperature counts as a warmer day
class Solution:
    def dailyTemperatures(self, temperatures):
        answer = [0] * len(temperatures)
        stack = []
        for day, temp in enumerate(temperatures):
            while stack and temperatures[stack[-1]] <= temp:
                prev = stack.pop()
                answer[prev] = day - prev
            stack.append(day)
        return answer
