"""Best Time to Buy and Sell Stock (LeetCode 121)."""

import random


def _best_profit(prices: list[int]) -> int:
    # Scan from the right, tracking the highest price still ahead of each day
    best, top = 0, prices[-1]
    for i in range(len(prices) - 2, -1, -1):
        best = max(best, top - prices[i])
        top = max(top, prices[i])
    return best


def _large_case():
    # 10^5 random prices: an O(n²) "try every buy day and sell day" scan does ~5·10^9 steps
    rng = random.Random(121)
    prices = [rng.randint(0, 10000) for _ in range(100000)]
    return {"args": [prices], "expected": _best_profit(prices)}


def _random_case(rng):
    return {"args": [[rng.randint(0, 9) for _ in range(rng.randint(1, 8))]]}


TASK = {
    "title": "Best Time to Buy and Sell Stock",
    "title_zh": "买卖股票的最佳时机",
    "difficulty": "Easy",
    "leetcode_id": 121,
    "leetcode_slug": "best-time-to-buy-and-sell-stock",
    "category": "Greedy",
    "function_name": "maxProfit",
    "params": [("prices", "int[]")],
    "returns": "int",
    "description_en": "The share price of one stock was recorded every day in `prices`, where `prices[i]` is the price on day `i`. You may buy once and sell once, with the sale on a later day than the purchase. Return the largest gain you can lock in. If every possible trade loses money, return `0`.\n\n**Example 1:** `prices = [3,8,2,6,1,5]` → `5` (buy at 3, sell at 8)\n\n**Example 2:** `prices = [9,4,4,2]` → `0` (the price only goes down)\n\n**Constraints:**\n- $1 \\le \\texttt{prices.length} \\le 10^{5}$\n- $0 \\le \\texttt{prices[i]} \\le 10^{4}$",
    "description_zh": "某只股票每天的价格记录在 `prices` 中，`prices[i]` 为第 `i` 天的价格。你只能买入一次、卖出一次，且卖出日必须晚于买入日。返回能锁定的最大收益；如果任何买卖都会亏损，返回 `0`。\n\n**示例 1：** `prices = [3,8,2,6,1,5]` → `5`（价格为 3 时买入，价格为 8 时卖出）\n\n**示例 2：** `prices = [9,4,4,2]` → `0`（价格一路下跌）\n\n**约束：**\n- $1 \\le \\texttt{prices.length} \\le 10^{5}$\n- $0 \\le \\texttt{prices[i]} \\le 10^{4}$",
    "hint": "Walk the days in order and keep the lowest price seen so far. The best sale on each day is that day's price minus this running minimum.",
    "hint_zh": "按天顺序遍历，记录到目前为止的最低价。每一天最好的卖出收益，就是当天价格减去这个最低价。",
    "cases": [
        {"args": [[3, 8, 2, 6, 1, 5]], "expected": 5},
        {"args": [[9, 4, 4, 2]], "expected": 0},
        {"args": [[1]], "expected": 0},
        # max - min gives 3 here, but the 4 comes before the 1, so the best trade is 2 -> 4
        {"args": [[2, 4, 1]], "expected": 2},
        {"args": [[8, 2, 9, 1, 4]], "expected": 7},
        {"args": [[5, 5, 5]], "expected": 0},
        {"args": [[0, 10000]], "expected": 10000},
        # The minimum appears twice; buying at the later copy is no worse than the first
        {"args": [[6, 1, 4, 1, 3]], "expected": 3},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        best, low = 0, prices[0]  # best gain so far, lowest price so far
        for p in prices:
            best = max(best, p - low)  # sell today, bought at the lowest earlier price
            low = min(low, p)
        return best''',
    "explanation": "Sell on each day at the cheapest price bought before it. One pass tracks the lowest price seen so far and the best profit. Sorting the trades by sell day is not needed, because the buy always comes first in the scan.\n\nTime O(n), space O(1).",
    "explanation_zh": "对每一天来说，最优的买入价就是它之前出现过的最低价。一次遍历同时维护到目前为止的最低价和最大收益。扫描时买入总是先于卖出，因此无需额外排序。\n\n时间 O(n)，空间 O(1)。",
}
