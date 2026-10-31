# Mistake: parses numbers with float(), which also accepts 1e3, 1_000, inf and nan
import re


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
    # BUG: float() accepts far more than plain decimals
    try:
        return float(s)
    except ValueError:
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
    return 0.1
