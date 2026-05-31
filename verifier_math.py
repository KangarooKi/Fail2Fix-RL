import math
import re

try:
    import sympy as sp
except Exception:
    sp = None

BOX_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL | re.IGNORECASE)
GSM8K_HASH_RE = re.compile(r"####\s*([^\n\r]+)")
ANSWER_LINE_RE = re.compile(r"(?:^|\n)\s*(?:final\s+answer|answer|答案)\s*[:：]\s*([^\n\r]+)", re.IGNORECASE)


def strip_latex_wrappers(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"<\|[^|]+?\|>", "", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("$", "")
    text = text.replace("\\,", "").replace("\\;", "")
    text = text.replace("\\!", "")
    return text.strip()


def normalize_text(text: str) -> str:
    text = strip_latex_wrappers(text)
    text = text.lower().strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("−", "-")
    text = text.replace("\\leq", "<=").replace("\\geq", ">=")
    return text


def extract_boxed(text: str):
    matches = BOX_RE.findall(str(text or ""))
    if matches:
        return matches[-1].strip()
    return None


def extract_answer_tag(text: str):
    matches = FINAL_ANSWER_RE.findall(str(text or "")) + ANSWER_RE.findall(str(text or ""))
    if matches:
        return matches[-1].strip()
    return None


def extract_answer_line(text: str):
    matches = ANSWER_LINE_RE.findall(str(text or ""))
    if matches:
        return matches[-1].strip()
    return None


def extract_gsm8k_hash_answer(text: str):
    matches = GSM8K_HASH_RE.findall(str(text or ""))
    if matches:
        return matches[-1].strip()
    return None


def extract_choice(text: str):
    text = str(text or "").strip()
    tail = text[-500:]
    patterns = [
        r"(?:answer|答案|选项)\s*(?:is|:|：)?\s*([ABCD])",
        r"\\boxed\s*\{\s*([ABCD])\s*\}",
        r"\b([ABCD])\b\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, tail, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def extract_final_answer(text: str) -> str:
    text = str(text or "")
    for fn in (extract_answer_tag, extract_answer_line, extract_gsm8k_hash_answer, extract_boxed):
        ans = fn(text)
        if ans:
            return ans
    choice = extract_choice(text)
    if choice:
        return choice
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    cue_patterns = [r"final answer", r"therefore", r"答案", r"所以", r"answer"]
    for line in reversed(lines):
        low = line.lower()
        if any(re.search(p, low) for p in cue_patterns):
            if ":" in line:
                return line.split(":")[-1].strip()
            if "：" in line:
                return line.split("：")[-1].strip()
            return line.strip()
    return lines[-1]


def answer_candidates(text: str):
    candidates = []
    final = extract_final_answer(text)
    raw = str(text or "").strip()
    for cand in (final, raw):
        if cand and cand not in candidates:
            candidates.append(cand)
        if cand and "=" in cand:
            rhs = cand.split("=")[-1].strip()
            if rhs and rhs not in candidates:
                candidates.append(rhs)
    return candidates


def latex_frac_to_sympy(expr: str) -> str:
    expr = strip_latex_wrappers(expr)
    expr = expr.replace("\\%", "").replace("%", "")
    expr = expr.replace("^\\circ", "*pi/180").replace("\\circ", "*pi/180")
    expr = re.sub(r"(\d+)\s*\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1+(\2)/(\3)", expr)
    expr = re.sub(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", expr)
    expr = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", expr)
    expr = expr.replace("^", "**")
    expr = expr.replace("\\pi", "pi")
    expr = expr.replace("×", "*").replace("·", "*")
    return expr


def parse_expr(expr: str):
    if sp is None:
        return None
    expr = latex_frac_to_sympy(expr).strip()
    if not expr:
        return None
    try:
        return sp.sympify(expr)
    except Exception:
        return None


def parse_interval(text: str):
    s = normalize_text(text)
    m = re.search(r"([\[\(])\s*([^,\[\]\(\)]+)\s*,\s*([^,\[\]\(\)]+)\s*([\]\)])", s)
    if not m:
        return None
    left_closed = m.group(1) == "["
    right_closed = m.group(4) == "]"
    left = parse_expr(m.group(2))
    right = parse_expr(m.group(3))
    if left is None or right is None:
        return None
    return (left_closed, left, right, right_closed)


def intervals_equivalent(a: str, b: str) -> bool:
    ia = parse_interval(a)
    ib = parse_interval(b)
    if ia is None or ib is None:
        return False
    if ia[0] != ib[0] or ia[3] != ib[3]:
        return False
    try:
        return bool(sp.simplify(ia[1] - ib[1]) == 0 and sp.simplify(ia[2] - ib[2]) == 0)
    except Exception:
        return False


def pair_equivalent(pred: str, ref: str) -> bool:
    pred_n = normalize_text(pred)
    ref_n = normalize_text(ref)
    if not pred_n or not ref_n:
        return False
    if pred_n == ref_n:
        return True
    if intervals_equivalent(pred, ref):
        return True
    pred_choice = extract_choice(pred)
    ref_choice = extract_choice(ref)
    if pred_choice and ref_choice and pred_choice == ref_choice:
        return True
    p_expr = parse_expr(pred)
    r_expr = parse_expr(ref)
    if p_expr is not None and r_expr is not None:
        try:
            if bool(sp.simplify(p_expr - r_expr) == 0):
                return True
            return abs(float(p_expr.evalf()) - float(r_expr.evalf())) <= 1e-2
        except Exception:
            try:
                return bool(p_expr.equals(r_expr))
            except Exception:
                return False
    return False


def equivalent(pred: str, ref: str) -> bool:
    for p in answer_candidates(pred):
        for r in answer_candidates(ref):
            if pair_equivalent(p, r):
                return True
    return False


def verify_math_response(response: str, reference_answer: str):
    pred = extract_final_answer(response)
    ref_final = extract_final_answer(reference_answer)
    ok = equivalent(pred, reference_answer)
    return {
        "reward": int(ok),
        "predicted_answer": pred,
        "reference_answer": reference_answer,
        "reference_final_answer": ref_final,
    }
