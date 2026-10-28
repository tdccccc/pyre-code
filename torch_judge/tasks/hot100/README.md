# Hot 100 problems: authoring guide and progress

Each file here defines one LeetCode Hot 100 problem as plain data. Both answer modes
(core code and ACM), the starters, the ACM input/output format text, the ACM reference
program and the UI case lists are generated from it by `torch_judge/algo/spec.py`.

## Writing a problem

Copy an existing file of the same shape and fill in:

- `two_sum.py`: a method; `move_zeroes.py`: in-place
- `reverse_linked_list.py` / `binary_tree_level_order.py`: linked lists / trees
- `lru_cache.py`: design class with arguments; `min_stack.py`: design with no-argument methods
- `intersection_of_two_linked_lists.py`: `inputs` + `prepare` / `finish` hooks, nullable answer
- `longest_palindromic_substring.py`: `custom` checker (several valid answers)

Keys:

| Key | Notes |
| --- | --- |
| `title`, `title_zh`, `leetcode_id`, `leetcode_slug` | Official titles and slug (used for the "view on LeetCode" link) |
| `difficulty` | `Easy` / `Medium` / `Hard`, as on LeetCode |
| `category` | English name from `CATEGORIES` in `algo/spec.py` (the study-plan section) |
| `function_name` | LeetCode's camelCase method name, or the class name for design problems |
| `params`, `returns` | Types below. In-place problems: `"returns": "void", "inplace": "<param>"` |
| `design` | Instead of params/returns: `{"class", "init": {name: type}, "methods": {name: ({name: type}, ret)}}` |
| `check` | `exact` (default), `unordered` (outer list in any order), `unordered_nested` (outer and inner lists in any order), `float` (1e-5 tolerance), `custom` (with `checker`) |
| `checker` | Source string `def check(args, actual, expected) -> bool`, for `check: "custom"`. `args` are the plain case args, `actual` the plain answer |
| `inputs` | Optional. The plain data a case's `args` hold, as `[(name, type)]`, when it differs from `params` (160: two value lists plus where they join). Drives the display, ACM input and ACM reading. Needs `prepare` |
| `prepare` | Source string `def prepare(args) -> list`: the `inputs` (already converted, e.g. `TreeNode` built) → the method's arguments |
| `finish` | Source string `def finish(result, args)`: the method's result (and its runtime args) → the plain answer, e.g. a node → its value. Needs `output` when the answer type differs from `returns` |
| `output` | Optional. Type of the plain answer when `finish` changes it. A trailing `?` (`int?`) allows `null` |
| `description_en/zh` | **Written in our own words with our own examples. Never copy LeetCode's text.** Format: statement, `**Example 1:** ... → ...`, `**Constraints:**` bullets, optional `**Follow-up:**` |
| `hint`, `hint_zh` | One or two sentences pointing at the key idea |
| `cases` | Hand-written. The first two are the visible samples that Run executes. Cover each classic pitfall with a comment saying which bug it catches. `"acm": False` for cases the ACM text format can't express (empty strings in output lists, whitespace-only lines) |
| `random_case(rng)` | Returns args only (`{"args": [...]}` or `{"ops", "args"}`); the reference solution fills in `expected`. Keep inputs small (they're shown in the UI). If `check` is `exact`/`unordered`, the answer must be unique (retry until it is, see `two_sum.py`). Must be ACM-safe |
| `large_case()` | Hidden, at the constraint limits, computes its own `expected`. Size it so the intended complexity passes easily and the naive one exceeds 3 s |
| `solution` | Clean core-mode reference (`class Solution` or the design class) |
| `explanation`, `explanation_zh` | Markdown: approach, why it works, `Time O(..), space O(..)` |

**Types:** `int`, `float`, `bool`, `str`, `int[]`, `int[][]`, `str[]`, `str[][]`,
`char[][]` (grids: `[["1","0"],...]`; ACM rows are written without spaces), `ListNode`,
`ListNode[]`, `TreeNode`, `Node` (list node with a `random` pointer, written
`[[val, random_index or null], ...]`), `void`. Lists and trees are written LeetCode-style
(`[1,2,null,3]`).

**Hooks** (`prepare`, `finish`, `checker`) are source strings because the sandboxed runner
can't import task modules. They can use `ListNode`, `TreeNode`, `Node`, `build_list`,
`list_to_array`, `build_tree`, `tree_to_array`, `build_random_list`, `random_list_to_array`
(same names in the generated ACM program).

**Design problems:** methods may take no arguments (`({}, "int")`); `str` arguments must not
contain spaces; `float` returns are compared with a tolerance in both modes.

## Checklist for every new problem

1. `python scripts/export_problems.py && python scripts/build_solutions.py`
2. `python -m pytest -q scripts` — the reference must pass every case in both modes
3. Add a classic wrong solution as `scripts/hot100_mutants/<task_id>.py`, starting with a
   `# Mistake: ...` line. At least 4 of the 20 random cases must catch it. If they don't, retune `random_case`
   (see the note in `lru_cache.py`)
4. Read the generated ACM format text and ACM program once (`spec.acm_format`, `spec.acm_solution`)

## Problems (100 / 100)

Order and sections follow the LeetCode 热题 100 study plan. All are in.

- 哈希 Hashing: [x] 1 两数之和 · [x] 49 字母异位词分组 · [x] 128 最长连续序列
- 双指针 Two Pointers: [x] 283 移动零 · [x] 11 盛最多水的容器 · [x] 15 三数之和 · [x] 42 接雨水
- 滑动窗口 Sliding Window: [x] 3 无重复字符的最长子串 · [x] 438 找到字符串中所有字母异位词
- 子串 Substring: [x] 560 和为 K 的子数组 · [x] 239 滑动窗口最大值 · [x] 76 最小覆盖子串
- 普通数组 Arrays: [x] 53 最大子数组和 · [x] 56 合并区间 · [x] 189 轮转数组 · [x] 238 除自身以外数组的乘积 · [x] 41 缺失的第一个正数
- 矩阵 Matrix: [x] 73 矩阵置零 · [x] 54 螺旋矩阵 · [x] 48 旋转图像 · [x] 240 搜索二维矩阵 II
- 链表 Linked List: [x] 160 相交链表 · [x] 206 反转链表 · [x] 234 回文链表 · [x] 141 环形链表 · [x] 142 环形链表 II · [x] 21 合并两个有序链表 · [x] 2 两数相加 · [x] 19 删除链表的倒数第 N 个结点 · [x] 24 两两交换链表中的节点 · [x] 25 K 个一组翻转链表 · [x] 138 随机链表的复制 · [x] 148 排序链表 · [x] 23 合并 K 个升序链表 · [x] 146 LRU 缓存
- 二叉树 Binary Tree: [x] 94 二叉树的中序遍历 · [x] 104 二叉树的最大深度 · [x] 226 翻转二叉树 · [x] 101 对称二叉树 · [x] 543 二叉树的直径 · [x] 102 二叉树的层序遍历 · [x] 108 将有序数组转换为二叉搜索树 · [x] 98 验证二叉搜索树 · [x] 230 二叉搜索树中第 K 小的元素 · [x] 199 二叉树的右视图 · [x] 114 二叉树展开为链表 · [x] 105 从前序与中序遍历序列构造二叉树 · [x] 437 路径总和 III · [x] 236 二叉树的最近公共祖先 · [x] 124 二叉树中的最大路径和
- 图论 Graphs: [x] 200 岛屿数量 · [x] 994 腐烂的橘子 · [x] 207 课程表 · [x] 208 实现 Trie (前缀树)
- 回溯 Backtracking: [x] 46 全排列 · [x] 78 子集 · [x] 17 电话号码的字母组合 · [x] 39 组合总和 · [x] 22 括号生成 · [x] 79 单词搜索 · [x] 131 分割回文串 · [x] 51 N 皇后
- 二分查找 Binary Search: [x] 35 搜索插入位置 · [x] 74 搜索二维矩阵 · [x] 34 在排序数组中查找元素的第一个和最后一个位置 · [x] 33 搜索旋转排序数组 · [x] 153 寻找旋转排序数组中的最小值 · [x] 4 寻找两个正序数组的中位数
- 栈 Stack: [x] 20 有效的括号 · [x] 155 最小栈 · [x] 394 字符串解码 · [x] 739 每日温度 · [x] 84 柱状图中最大的矩形
- 堆 Heap: [x] 215 数组中的第K个最大元素 · [x] 347 前 K 个高频元素 · [x] 295 数据流的中位数
- 贪心算法 Greedy: [x] 121 买卖股票的最佳时机 · [x] 55 跳跃游戏 · [x] 45 跳跃游戏 II · [x] 763 划分字母区间
- 动态规划 Dynamic Programming: [x] 70 爬楼梯 · [x] 118 杨辉三角 · [x] 198 打家劫舍 · [x] 279 完全平方数 · [x] 322 零钱兑换 · [x] 139 单词拆分 · [x] 300 最长递增子序列 · [x] 152 乘积最大子数组 · [x] 416 分割等和子集 · [x] 32 最长有效括号
- 多维动态规划 Multi-dimensional DP: [x] 62 不同路径 · [x] 64 最小路径和 · [x] 5 最长回文子串 · [x] 1143 最长公共子序列 · [x] 72 编辑距离
- 技巧 Techniques: [x] 136 只出现一次的数字 · [x] 169 多数元素 · [x] 75 颜色分类 · [x] 31 下一个排列 · [x] 287 寻找重复数

Notes on how the trickier ones are expressed (useful when revising them):

- Checkers: 15, 39, 78 use `unordered_nested`; 46, 131, 17, 22, 51, 347, 438 use `unordered`;
  4 uses `float`; 5, 76, 108 use `custom`
- Hooks: 141 / 142 build the cycle from `pos` (142 reports the entry node by index); 160 joins
  two lists at a shared tail; 236 picks p / q by value and reports the answer node's value;
  138's `finish` rejects copies that reuse original nodes
- Recursion: both modes run with a recursion limit of 20000 (ACM programs are started through
  a small bootstrap that raises it), so keep large trees and lists under ~10^4 levels
- ACM nested answers (`int[][]`, `str[][]`) may end with an empty row; the judge retries the
  parse keeping trailing blank lines before calling it wrong
