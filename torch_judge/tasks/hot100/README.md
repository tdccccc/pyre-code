# Hot 100 problems: authoring guide and progress

Each file here defines one LeetCode Hot 100 problem as plain data. Both answer modes
(core code and ACM), the starters, the ACM input/output format text, the ACM reference
program and the UI case lists are generated from it by `torch_judge/algo/spec.py`.

## Writing a problem

Copy an existing file of the same shape (`two_sum.py` for a method, `lru_cache.py` for a
design class, `move_zeroes.py` for in-place, `reverse_linked_list.py` /
`binary_tree_level_order.py` for linked lists / trees) and fill in:

| Key | Notes |
| --- | --- |
| `title`, `title_zh`, `leetcode_id`, `leetcode_slug` | Official titles and slug (used for the "view on LeetCode" link) |
| `difficulty` | `Easy` / `Medium` / `Hard`, as on LeetCode |
| `category` | English name from `CATEGORIES` in `algo/spec.py` (the study-plan section) |
| `function_name` | LeetCode's camelCase method name, or the class name for design problems |
| `params`, `returns` | Types below. In-place problems: `"returns": "void", "inplace": "<param>"` |
| `design` | Instead of params/returns: `{"class", "init": {name: type}, "methods": {name: ({name: type}, ret)}}` |
| `check` | `exact` (default), `unordered`, `unordered_nested`, `float` |
| `description_en/zh` | **Written in our own words with our own examples. Never copy LeetCode's text.** Format: statement, `**Example 1:** ... → ...`, `**Constraints:**` bullets, optional `**Follow-up:**` |
| `hint`, `hint_zh` | One or two sentences pointing at the key idea |
| `cases` | Hand-written. The first two are the visible samples that Run executes. Cover each classic pitfall with a comment saying which bug it catches. `"acm": False` for cases the ACM text format can't express (empty strings in output lists, whitespace-only lines) |
| `random_case(rng)` | Returns args only (`{"args": [...]}` or `{"ops", "args"}`); the reference solution fills in `expected`. Keep inputs small (they're shown in the UI). If `check` is `exact`/`unordered`, the answer must be unique (retry until it is, see `two_sum.py`). Must be ACM-safe |
| `large_case()` | Hidden, at the constraint limits, computes its own `expected`. Size it so the intended complexity passes easily and the naive one exceeds 3 s |
| `solution` | Clean core-mode reference (`class Solution` or the design class) |
| `explanation`, `explanation_zh` | Markdown: approach, why it works, `Time O(..), space O(..)` |

**Types:** `int`, `float`, `bool`, `str`, `int[]`, `int[][]`, `str[]`, `str[][]`, `ListNode`,
`TreeNode`, `void`. Lists and trees are written LeetCode-style (`[1,2,null,3]`).

## Checklist for every new problem

1. `python scripts/export_problems.py && python scripts/build_solutions.py`
2. `python -m pytest -q scripts` — the reference must pass every case in both modes
3. Add a classic wrong solution to `MUTANTS` in `scripts/test_hot100.py`; the random cases must catch it.
   If they don't, retune `random_case` (see the note in `lru_cache.py`)
4. Read the generated ACM format text and ACM program once (`spec.acm_format`, `spec.acm_solution`)

## Framework extensions still needed

Do these before the problems that depend on them:

- **`char[][]`** grids (200 岛屿数量, 79 单词搜索): core `List[List[str]]`; ACM rows as strings without spaces
- **`ListNode[]`** (23 合并 K 个升序链表): ACM `k`, then `n` + values per list
- **Custom builders** (as source strings, since the runner can't import task modules):
  `prepare(args) -> runtime args` and `finish(result, runtime_args) -> plain value`.
  Needed by 160 相交链表 (shared tail), 141/142 环形链表 (`pos` cycle; 142 returns the entry node → its index),
  138 随机链表的复制 (random pointers; also check the copy shares no nodes), 236 最近公共祖先 (p/q values → nodes; result node → value)
- **Custom checkers** (source string `check(args, actual, expected) -> bool`) where several answers are valid:
  5 最长回文子串, 108 有序数组转二叉搜索树
- **Design problems with no-argument methods and `float` returns** (155 最小栈, 295 数据流的中位数),
  plus `str` args (208 Trie). ACM design output must compare floats numerically

## Progress (8 / 100)

Order and sections follow the LeetCode 热题 100 study plan. Checked = done.

- 哈希 Hashing: [x] 1 两数之和 · [x] 49 字母异位词分组 · [ ] 128 最长连续序列
- 双指针 Two Pointers: [x] 283 移动零 · [ ] 11 盛最多水的容器 · [ ] 15 三数之和 · [ ] 42 接雨水
- 滑动窗口 Sliding Window: [x] 3 无重复字符的最长子串 · [ ] 438 找到字符串中所有字母异位词
- 子串 Substring: [ ] 560 和为 K 的子数组 · [ ] 239 滑动窗口最大值 · [ ] 76 最小覆盖子串
- 普通数组 Arrays: [ ] 53 最大子数组和 · [ ] 56 合并区间 · [ ] 189 轮转数组 · [ ] 238 除自身以外数组的乘积 · [ ] 41 缺失的第一个正数
- 矩阵 Matrix: [ ] 73 矩阵置零 · [ ] 54 螺旋矩阵 · [ ] 48 旋转图像 · [ ] 240 搜索二维矩阵 II
- 链表 Linked List: [ ] 160 相交链表 · [x] 206 反转链表 · [ ] 234 回文链表 · [ ] 141 环形链表 · [ ] 142 环形链表 II · [ ] 21 合并两个有序链表 · [ ] 2 两数相加 · [ ] 19 删除链表的倒数第 N 个结点 · [ ] 24 两两交换链表中的节点 · [ ] 25 K 个一组翻转链表 · [ ] 138 随机链表的复制 · [ ] 148 排序链表 · [ ] 23 合并 K 个升序链表 · [x] 146 LRU 缓存
- 二叉树 Binary Tree: [ ] 94 二叉树的中序遍历 · [ ] 104 二叉树的最大深度 · [ ] 226 翻转二叉树 · [ ] 101 对称二叉树 · [ ] 543 二叉树的直径 · [x] 102 二叉树的层序遍历 · [ ] 108 将有序数组转换为二叉搜索树 · [ ] 98 验证二叉搜索树 · [ ] 230 二叉搜索树中第 K 小的元素 · [ ] 199 二叉树的右视图 · [ ] 114 二叉树展开为链表 · [ ] 105 从前序与中序遍历序列构造二叉树 · [ ] 437 路径总和 III · [ ] 236 二叉树的最近公共祖先 · [ ] 124 二叉树中的最大路径和
- 图论 Graphs: [ ] 200 岛屿数量 · [ ] 994 腐烂的橘子 · [ ] 207 课程表 · [ ] 208 实现 Trie (前缀树)
- 回溯 Backtracking: [ ] 46 全排列 · [ ] 78 子集 · [ ] 17 电话号码的字母组合 · [ ] 39 组合总和 · [ ] 22 括号生成 · [ ] 79 单词搜索 · [ ] 131 分割回文串 · [ ] 51 N 皇后
- 二分查找 Binary Search: [ ] 35 搜索插入位置 · [ ] 74 搜索二维矩阵 · [ ] 34 在排序数组中查找元素的第一个和最后一个位置 · [ ] 33 搜索旋转排序数组 · [ ] 153 寻找旋转排序数组中的最小值 · [ ] 4 寻找两个正序数组的中位数
- 栈 Stack: [ ] 20 有效的括号 · [ ] 155 最小栈 · [ ] 394 字符串解码 · [ ] 739 每日温度 · [ ] 84 柱状图中最大的矩形
- 堆 Heap: [ ] 215 数组中的第K个最大元素 · [ ] 347 前 K 个高频元素 · [ ] 295 数据流的中位数
- 贪心算法 Greedy: [ ] 121 买卖股票的最佳时机 · [ ] 55 跳跃游戏 · [ ] 45 跳跃游戏 II · [ ] 763 划分字母区间
- 动态规划 Dynamic Programming: [ ] 70 爬楼梯 · [ ] 118 杨辉三角 · [ ] 198 打家劫舍 · [ ] 279 完全平方数 · [x] 322 零钱兑换 · [ ] 139 单词拆分 · [ ] 300 最长递增子序列 · [ ] 152 乘积最大子数组 · [ ] 416 分割等和子集 · [ ] 32 最长有效括号
- 多维动态规划 Multi-dimensional DP: [ ] 62 不同路径 · [ ] 64 最小路径和 · [ ] 5 最长回文子串 · [ ] 1143 最长公共子序列 · [ ] 72 编辑距离
- 技巧 Techniques: [ ] 136 只出现一次的数字 · [ ] 169 多数元素 · [ ] 75 颜色分类 · [ ] 31 下一个排列 · [ ] 287 寻找重复数

Checker notes for later problems: 15 三数之和 and 39 组合总和 / 78 子集 use `unordered_nested`;
46 全排列, 131 分割回文串, 17, 22, 51, 347, 438 use `unordered` (inner order matters or items are strings);
4 and 295 return floats.
