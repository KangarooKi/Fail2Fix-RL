import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
from modelscope.msdatasets import MsDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import sympy as sp
except Exception:
    sp = None


BOX_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def strip_latex(text):
    text = str(text or "").strip()
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\;": "",
        "\\!": "",
        "\\cdot": "*",
        "\\times": "*",
        "×": "*",
        "−": "-",
        "$": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()


def extract_boxed(text):
    matches = BOX_RE.findall(str(text or ""))
    return matches[-1].strip() if matches else None


def extract_final_answer(text):
    text = str(text or "")
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    tag = ANSWER_TAG_RE.findall(text)
    if tag:
        boxed_in_tag = extract_boxed(tag[-1])
        return boxed_in_tag or tag[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    cue = re.compile(r"(final answer|answer is|therefore|thus|答案|所以)", re.IGNORECASE)
    for line in reversed(lines[-12:]):
        if cue.search(line):
            boxed = extract_boxed(line)
            if boxed:
                return boxed
            if ":" in line:
                return simplify_answer_line(line.split(":")[-1].strip())
            if "：" in line:
                return simplify_answer_line(line.split("：")[-1].strip())
            return simplify_answer_line(line)
    return simplify_answer_line(lines[-1])


def simplify_answer_line(line):
    line = str(line or "").replace("<|im_end|>", "").strip()
    boxed = extract_boxed(line)
    if boxed:
        return boxed
    math_spans = re.findall(r"\\\((.*?)\\\)|\$(.*?)\$", line)
    flat_spans = [a or b for a, b in math_spans if (a or b).strip()]
    if flat_spans:
        return flat_spans[-1].strip()
    frac = re.findall(r"\\(?:dfrac|tfrac|frac)\s*\{[^{}]+\}\s*\{[^{}]+\}", line)
    if frac:
        return frac[-1].strip()
    tuple_like = re.findall(r"[\(\[]\s*[-+\\\w{}^/. ]+\s*,\s*[-+\\\w{}^/. ]+\s*[\)\]]", line)
    if tuple_like:
        return tuple_like[-1].strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?", line)
    if numbers:
        return numbers[-1].strip()
    return line


def normalize(text):
    text = strip_latex(text).lower()
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\leq", "<=").replace("\\geq", ">=")
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".")
    return text


def latex_to_sympy(text):
    text = strip_latex(text)
    text = text.replace("\\%", "/100").replace("%", "/100")
    text = re.sub(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = text.replace("^", "**")
    text = text.replace("\\pi", "pi")
    text = text.replace("\\infty", "oo")
    return text


def top_level_split(text, sep=","):
    parts = []
    cur = []
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return parts


def unwrap_outer(text):
    text = strip_latex(text).strip()
    pairs = {"(": ")", "[": "]", "{": "}"}
    if len(text) >= 2 and text[0] in pairs and text[-1] == pairs[text[0]]:
        depth = 0
        for i, ch in enumerate(text):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if depth == 0 and i != len(text) - 1:
                return text
        return text[1:-1].strip()
    return text


def parse_expr(text):
    if sp is None:
        return None
    text = latex_to_sympy(text).strip()
    if not text:
        return None
    try:
        return sp.sympify(text)
    except Exception:
        return None


def expr_equiv(a, b):
    ae = parse_expr(a)
    be = parse_expr(b)
    if ae is None or be is None:
        return False
    try:
        if bool(sp.simplify(ae - be) == 0):
            return True
    except Exception:
        pass
    try:
        return math.isclose(float(ae.evalf()), float(be.evalf()), rel_tol=1e-5, abs_tol=1e-5)
    except Exception:
        return False


def list_equiv(pred, ref):
    pu = unwrap_outer(pred)
    ru = unwrap_outer(ref)
    if "," not in pu or "," not in ru:
        return False
    pp = top_level_split(pu)
    rr = top_level_split(ru)
    if len(pp) != len(rr):
        return False
    return all(equivalent(p, r) for p, r in zip(pp, rr))


def equivalent(pred, ref):
    pred = str(pred or "").strip()
    ref = str(ref or "").strip()
    if not pred or not ref:
        return False
    pred_rhs = assignment_rhs(pred)
    ref_rhs = assignment_rhs(ref)
    if (pred_rhs != pred or ref_rhs != ref) and equivalent(pred_rhs, ref_rhs):
        return True
    if normalize(pred) == normalize(ref):
        return True
    if list_equiv(pred, ref):
        return True
    if expr_equiv(pred, ref):
        return True
    pred_box = extract_boxed(pred)
    ref_box = extract_boxed(ref)
    if pred_box or ref_box:
        return equivalent(pred_box or pred, ref_box or ref)
    return False


def assignment_rhs(text):
    text = str(text or "").strip()
    if "=" not in text or any(op in text for op in ("<=", ">=", "\\le", "\\ge")):
        return text
    parts = text.split("=")
    rhs = parts[-1].strip()
    return rhs or text


def prompt_text(tokenizer, problem):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": str(problem).strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_math500(limit=None):
    ds = MsDataset.load("AI-ModelScope/MATH-500", split="test")
    rows = []
    for i, row in enumerate(ds):
        if limit is not None and len(rows) >= limit:
            break
        rows.append(
            {
                "idx": i,
                "unique_id": row.get("unique_id", str(i)),
                "problem": row["problem"],
                "answer": row["answer"],
                "solution": row.get("solution", ""),
                "subject": row.get("subject", ""),
                "level": row.get("level", None),
            }
        )
    return rows


def load_rows_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "idx": row.get("idx", len(rows)),
                    "unique_id": row.get("unique_id", str(len(rows))),
                    "problem": row["problem"],
                    "answer": row["answer"],
                    "solution": row.get("solution", ""),
                    "subject": row.get("subject", ""),
                    "level": row.get("level", None),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def read_done(path):
    done = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                done[item["unique_id"]] = item
    return done


def summarize(items):
    n = len(items)
    correct = sum(int(x["correct"]) for x in items)
    by_subject = {}
    by_level = {}
    for x in items:
        for group, key in ((by_subject, x.get("subject") or "unknown"), (by_level, str(x.get("level")))):
            cur = group.setdefault(key, {"count": 0, "correct": 0})
            cur["count"] += 1
            cur["correct"] += int(x["correct"])
    for group in (by_subject, by_level):
        for cur in group.values():
            cur["accuracy"] = cur["correct"] / max(1, cur["count"])
    return {
        "count": n,
        "correct": correct,
        "accuracy": correct / max(1, n),
        "empty_answer_rate": sum(1 for x in items if not str(x.get("predicted_answer", "")).strip()) / max(1, n),
        "truncated_rate": sum(1 for x in items if x.get("truncated")) / max(1, n),
        "avg_new_tokens": sum(x.get("new_tokens", 0) for x in items) / max(1, n),
        "by_subject": by_subject,
        "by_level": by_level,
    }


def evaluate_model(model_name, model_path, rows, args):
    out_path = Path(args.output_dir) / f"math500_{model_name}_n{len(rows)}_tok{args.max_new_tokens}.jsonl"
    done = read_done(out_path) if args.resume else {}

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    items = list(done.values())
    with out_path.open("a", encoding="utf-8") as out:
        for pos, row in enumerate(rows, start=1):
            if row["unique_id"] in done:
                continue
            enc = tokenizer(
                prompt_text(tokenizer, row["problem"]),
                return_tensors="pt",
                truncation=True,
                max_length=args.prompt_max_length,
            )
            enc.pop("token_type_ids", None)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            prompt_len = enc["input_ids"].shape[1]
            start = time.time()
            with torch.inference_mode():
                gen = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.time() - start
            new_tokens = gen.shape[1] - prompt_len
            completion = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=False).strip()
            pred = extract_final_answer(completion)
            correct = equivalent(pred, row["answer"])
            item = {
                **row,
                "model_name": model_name,
                "model_path": model_path,
                "completion": completion,
                "predicted_answer": pred,
                "correct": bool(correct),
                "new_tokens": int(new_tokens),
                "truncated": bool(new_tokens >= args.max_new_tokens),
                "elapsed_sec": elapsed,
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()
            items.append(item)
            print(
                f"[{model_name}] {pos}/{len(rows)} acc={sum(int(x['correct']) for x in items)}/{len(items)} "
                f"tok={new_tokens} pred={pred[:80]!r} ref={row['answer'][:80]!r}",
                flush=True,
            )

    del model
    torch.cuda.empty_cache()
    return out_path, summarize(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="/root/models")
    parser.add_argument("--sft-model", default="/root/autodl-tmp/learning_from_failure_exp/checkpoints/qwen25_05b_sft_best")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--prompt-max-length", type=int, default=4096)
    parser.add_argument("--output-dir", default="/root/autodl-tmp/learning_from_failure_exp/reports/math500_bench")
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--only", choices=["base", "sft", "both"], default="both")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    limit = args.limit if args.limit > 0 else None
    rows = load_rows_jsonl(args.input_jsonl, limit) if args.input_jsonl else load_math500(limit)
    report = {
        "dataset": args.input_jsonl or "AI-ModelScope/MATH-500",
        "limit": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "prompt_max_length": args.prompt_max_length,
        "models": {},
    }
    if args.only in ("base", "both"):
        path, summary = evaluate_model("base", args.base_model, rows, args)
        report["models"]["base"] = {"path": args.base_model, "results_jsonl": str(path), **summary}
    if args.only in ("sft", "both"):
        path, summary = evaluate_model("sft", args.sft_model, rows, args)
        report["models"]["sft"] = {"path": args.sft_model, "results_jsonl": str(path), **summary}

    summary_path = Path(args.output_dir) / f"math500_summary_n{len(rows)}_tok{args.max_new_tokens}_{args.only}.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()
