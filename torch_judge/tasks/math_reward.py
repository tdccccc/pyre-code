"""Verifiable Math Reward task."""

TASK = {
    "title": "Verifiable Math Reward",
    "title_zh": "可验证数学奖励",
    "difficulty": "Medium",
    "description_en": "Implement the rule-based reward used to train reasoning models with RL on math problems (RLVR, as in DeepSeek-R1): check the output format, extract the final answer, and compare it with the reference answer.\n\n**Signature:** `math_reward(response: str, answer: str) -> float`\n\n**Parameters:**\n- `response` — the model's full output text\n- `answer` — the reference answer (a LaTeX or plain string)\n\n**Returns:** `1.0` if the format is valid and the answer is correct, `0.1` if the format is valid but the answer is wrong, `0.0` if the format is invalid\n\n**Format (all must hold, otherwise return 0.0):**\n- The response contains exactly one `<think>` and exactly one `</think>`.\n- Only whitespace may appear before `<think>` (so `</think>` cannot come first).\n- At least one `\\boxed{` appears after `</think>`. Boxes inside the think block do not count.\n\n**Extraction:** the prediction is the content of the **last** `\\boxed{...}` after `</think>`, matching nested braces (`\\boxed{\\frac{1}{2}}` gives `\\frac{1}{2}`). If that last box is never closed, the format is invalid.\n\n**Normalization (applied to both prediction and answer, in this order):**\n1. Remove all whitespace.\n2. Remove `$`, `\\left`, `\\right` and `\\!`.\n3. Replace `\\dfrac` and `\\tfrac` with `\\frac`.\n4. If the string ends with `.`, remove that one character (only one: `x..` becomes `x.`).\n5. If the **whole** string is a number with thousands separators (optional sign, then `\\d{1,3}(,\\d{3})+`, e.g. `1,000`), drop its commas. Commas anywhere else stay (`(1,234)` is unchanged).\n\n**Correctness:** correct if the normalized strings are equal, or if both parse as numbers that agree within a relative tolerance: $|p - q| \\le 10^{-6} \\cdot \\max(1, |q|)$, where $q$ is the reference answer. Number forms (each with an optional leading `+` or `-`): integer or decimal matching `\\d+(\\.\\d*)?` or `\\.\\d+` (`3`, `2.5`, `12.`, `.5`), `a/b`, and `\\frac{a}{b}`, with $a, b$ non-negative integers. Nothing else parses: no exponents (`1e3`), underscores (`1_000`), `inf` or `nan`. A zero denominator does not parse.",
    "description_zh": "实现数学题强化学习（RLVR，如 DeepSeek-R1）中使用的规则奖励：检查输出格式，抽取最终答案，并与标准答案比较。\n\n**签名:** `math_reward(response: str, answer: str) -> float`\n\n**参数:**\n- `response` — 模型的完整输出文本\n- `answer` — 标准答案（LaTeX 或普通字符串）\n\n**返回:** 格式正确且答案正确返回 `1.0`；格式正确但答案错误返回 `0.1`；格式错误返回 `0.0`\n\n**格式（必须全部满足，否则返回 0.0）:**\n- 回答中恰好有一个 `<think>` 和一个 `</think>`。\n- `<think>` 之前只能有空白字符（因此 `</think>` 不可能在前）。\n- `</think>` 之后至少出现一次 `\\boxed{`。思考块里的框不算。\n\n**抽取:** 预测答案是 `</think>` 之后**最后一个** `\\boxed{...}` 的内容，需要匹配嵌套的花括号（`\\boxed{\\frac{1}{2}}` 得到 `\\frac{1}{2}`）。如果最后一个框没有闭合，视为格式错误。\n\n**规范化（对预测和标准答案都做，按以下顺序）:**\n1. 删除所有空白字符。\n2. 删除 `$`、`\\left`、`\\right` 和 `\\!`。\n3. 把 `\\dfrac`、`\\tfrac` 替换为 `\\frac`。\n4. 如果字符串以 `.` 结尾，去掉这一个字符（只去一个：`x..` 变成 `x.`）。\n5. 如果**整个**字符串是带千位分隔符的数（可带符号，然后满足 `\\d{1,3}(,\\d{3})+`，如 `1,000`），去掉它的逗号。其他位置的逗号保留（`(1,234)` 不变）。\n\n**判对:** 规范化后的字符串相等，或者两者都能解析成数，且在相对误差内一致：$|p - q| \\le 10^{-6} \\cdot \\max(1, |q|)$，其中 $q$ 是标准答案。可解析的数（都可以带一个前导 `+` 或 `-`）：匹配 `\\d+(\\.\\d*)?` 或 `\\.\\d+` 的整数或小数（`3`、`2.5`、`12.`、`.5`）、`a/b`、`\\frac{a}{b}`，其中 $a, b$ 是非负整数。其他形式都不算数：没有科学计数法（`1e3`）、下划线（`1_000`）、`inf` 或 `nan`。分母为 0 时也不算能解析。",
    "function_name": "math_reward",
    "hint": "1. Format: count('<think>') == 1 and count('</think>') == 1, <think> before </think>, only whitespace before <think>\n2. tail = text after </think>; start = tail.rfind('\\\\boxed{'); walk forward counting { and } until depth hits 0\n3. normalize both strings (whitespace, $, \\\\left, \\\\right, \\\\!, dfrac/tfrac, trailing '.', thousands commas)\n4. equal strings -> 1.0; else parse both with regexes for int/decimal, a/b, \\\\frac{a}{b} and compare with relative tolerance\n5. otherwise 0.1",
    "hint_zh": "1. 格式：count('<think>') == 1 且 count('</think>') == 1，<think> 在 </think> 之前，<think> 前只有空白\n2. tail = </think> 之后的文本；start = tail.rfind('\\\\boxed{')；向后数 { 和 }，深度回到 0 时结束\n3. 两个字符串都规范化（空白、$、\\\\left、\\\\right、\\\\!、dfrac/tfrac、末尾的 '.'、千位逗号）\n4. 字符串相等 -> 1.0；否则用正则把两者解析成数（整数/小数、a/b、\\\\frac{a}{b}），按相对误差比较\n5. 其余情况 0.1",
    "tests": [
        {
            "name": "Correct, wrong, and return type",
            "code": r"""
ok = {fn}(r'<think>6 times 7</think> The answer is \boxed{42}.', '42')
assert isinstance(ok, float), f'Expected a float, got {type(ok).__name__}'
assert ok == 1.0, f'Correct answer should give 1.0, got {ok}'
wrong = {fn}(r'<think>6 times 7</think> The answer is \boxed{41}.', '42')
assert wrong == 0.1, f'Well-formatted wrong answer should give 0.1, got {wrong}'
lead = {fn}('\n  <think>x</think>' + r'\boxed{42}', '42')
assert lead == 1.0, f'Whitespace before <think> is allowed, got {lead}'
""",
        },
        {
            "name": "Format violations give 0.0",
            "code": r"""
cases = {
    'no think block': (r'The answer is \boxed{42}', '42'),
    'two <think> tags': (r'<think>a</think><think>b</think>\boxed{42}', '42'),
    'closing tag first': (r'</think>reason<think> \boxed{42}', '42'),
    'text before <think>': (r'Sure! <think>a</think>\boxed{42}', '42'),
    'box only inside think': (r'<think>so \boxed{42}</think> done', '42'),
    'no box after think': (r'<think>a</think> The answer is 42', '42'),
    'last box never closed': (r'<think>a</think> \boxed{41} then \boxed{42', '42'),
}
for name, (resp, ans) in cases.items():
    got = {fn}(resp, ans)
    assert got == 0.0, f'{name}: expected 0.0, got {got}'
""",
        },
        {
            "name": "Uses the last box after </think>",
            "code": r"""
resp = r'<think>try \boxed{7}</think> First \boxed{3}, wait, actually \boxed{5}.'
assert {fn}(resp, '5') == 1.0, 'The last box after </think> is the prediction'
assert {fn}(resp, '3') == 0.1, 'An earlier box must not be used'
assert {fn}(resp, '7') == 0.1, 'A box inside the think block must not be used'
""",
        },
        {
            "name": "Nested braces inside the box",
            "code": r"""
cases = [
    (r'\frac{1}{2}', r'\frac{1}{2}'),
    (r'\sqrt{2}', r'\sqrt{2}'),
    (r'x^{2}+1', r'x^{2} + 1'),
    (r'\frac{\sqrt{3}}{2}', r'\frac{\sqrt{3}}{2}'),
]
for boxed, ans in cases:
    resp = '<think>work</think> So \\boxed{' + boxed + '} is it.'
    got = {fn}(resp, ans)
    assert got == 1.0, f'\\boxed{{{boxed}}} vs {ans!r}: expected 1.0, got {got}'
""",
        },
        {
            "name": "Normalization rules",
            "code": r"""
def r(boxed, ans):
    return {fn}('<think>w</think> \\boxed{' + boxed + '}', ans)

same = [
    (r'$\dfrac{3}{4}$', r'\frac{3}{4}'),
    (r'\tfrac{1}{3}', r'\frac{1}{3}'),
    (r'\left( 1, 2 \right)', '(1,2)'),
    ('1,000', '1000'),
    ('-12,345,678', '-12345678'),
    ('12.', '12'),
    (r'3\!000', '3000'),
    (r'\{1, 2\}', r'\{1,2\}'),
    ('1, 000', '1000'),   # whitespace goes first, then the thousands check
    ('$x.$', 'x'),        # $ goes before the trailing-dot strip
    ('1\n2', '12'),
    ('1\t2', '12'),
]
for boxed, ans in same:
    assert r(boxed, ans) == 1.0, f'{boxed!r} should normalize to match {ans!r}'

different = [
    ('(1,2)', '(1,3)'),
    ('1,2', '12'),        # not a thousands pattern: the comma stays
    ('1,00', '100'),
    ('(1,234)', '(1234)'),  # commas are only dropped when the whole string is a number
    ('x..', 'x'),         # only one trailing dot is stripped
    ('x+1', 'x-1'),
]
for boxed, ans in different:
    assert r(boxed, ans) == 0.1, f'{boxed!r} must not match {ans!r}'
""",
        },
        {
            "name": "Numeric equivalence and tolerance",
            "code": r"""
def r(boxed, ans):
    return {fn}('<think>w</think> \\boxed{' + boxed + '}', ans)

same = [
    ('0.5', r'\frac{1}{2}'),
    ('1/2', '0.5'),
    (r'-\frac{3}{4}', '-0.75'),
    ('2.0', '2'),
    ('.25', '1/4'),
    ('+3', '3'),
    ('0.3333333', r'\frac{1}{3}'),
    ('1000000.5', '1000000'),     # relative tolerance: 1e-6 * 1e6 = 1
    ('0.0000015', '0.000001'),    # absolute floor: tolerance is 1e-6 * max(1, |q|)
]
for boxed, ans in same:
    assert r(boxed, ans) == 1.0, f'{boxed!r} should equal {ans!r} numerically'

different = [
    ('0.333', r'\frac{1}{3}'),
    ('1000002', '1000000'),
    ('0.000003', '0.000001'),
    ('3.14', r'\pi'),
    (r'\frac{1}{0}', '0'),
    ('1/0', '1/0.0'),
    ('1e3', '1000'),      # no exponents
    ('1_000', '1000'),    # no underscores
    ('inf', '1e309'),
]
for boxed, ans in different:
    assert r(boxed, ans) == 0.1, f'{boxed!r} must not equal {ans!r}'
""",
        },
        {
            "name": "Large and messy responses",
            "code": r"""
think = 'Let me compute. ' * 200 + r'Maybe \boxed{1}? {not closed'
resp = '<think>' + think + '</think>\n\nTherefore the area is $\\boxed{\\dfrac{9}{2}}$.'
assert {fn}(resp, '4.5') == 1.0, 'Unbalanced braces inside <think> must not affect the answer'
assert {fn}(resp, r'\frac{9}{2}') == 1.0, 'LaTeX reference answer should match too'
assert {fn}(resp + ' \\boxed{5}', '4.5') == 0.1, 'A later box replaces the earlier one'
""",
        },
    ],
    "solution": r'''import re


def _last_boxed(text):
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None  # the last box is never closed


def _normalize(s):
    s = re.sub(r"\s+", "", s)
    for tok in ("$", "\\left", "\\right", "\\!"):
        s = s.replace(tok, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    if s.endswith("."):
        s = s[:-1]
    if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    return s


def _to_number(s):
    m = re.fullmatch(r"([+-]?)\\frac\{(\d+)\}\{(\d+)\}", s) or re.fullmatch(r"([+-]?)(\d+)/(\d+)", s)
    if m is not None:
        sign, a, b = m.groups()
        if int(b) == 0:
            return None
        value = int(a) / int(b)
        return -value if sign == "-" else value
    if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)", s):
        return float(s)
    return None


def math_reward(response, answer):
    # Format: exactly one think block, at the start, followed by a boxed answer
    if response.count("<think>") != 1 or response.count("</think>") != 1:
        return 0.0
    open_at, close_at = response.find("<think>"), response.find("</think>")
    if open_at > close_at or response[:open_at].strip():
        return 0.0
    pred = _last_boxed(response[close_at + len("</think>"):])
    if pred is None:
        return 0.0

    pred, gold = _normalize(pred), _normalize(answer)
    if pred == gold:
        return 1.0
    p, q = _to_number(pred), _to_number(gold)
    if p is not None and q is not None and abs(p - q) <= 1e-6 * max(1.0, abs(q)):
        return 1.0
    return 0.1''',
    "demo": r"""responses = [
    (r'<think>2 + 2 = 4</think> The answer is \boxed{4}.', '4'),
    (r'<think>half of one</think> \boxed{\dfrac{1}{2}}', '0.5'),
    (r'<think>hmm</think> First \boxed{3}, no wait, \boxed{5}.', '3'),
    (r'The answer is \boxed{4}.', '4'),
    (r'<think>so \boxed{4}</think> I am done.', '4'),
]
for resp, ans in responses:
    print(f'{math_reward(resp, ans):.1f}  answer={ans:<5} {resp}')""",

}
