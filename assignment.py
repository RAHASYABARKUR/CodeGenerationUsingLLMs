#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, re, csv, json, argparse, random, traceback, multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
from datasets import load_dataset

# =================== Config ===================

DEFAULT_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "codellama/CodeLlama-7b-Instruct-hf",
]

# =================== Helpers ===================

def set_seed(seed=42):
    random.seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except Exception:
        pass

def load_humaneval(n=10):
    ds = load_dataset("openai_humaneval", split="test")
    return ds.select(range(n))

def infer_family(mid: str) -> str:
    l = mid.lower()
    if "mistral" in l: return "Mistral"
    if "llama" in l: return "LLaMA"
    return "Other"

# =================== Prompt helpers ===================

def strict_suffix(entry):
    return (f"Output ONLY one Python function named `{entry}` inside triple backticks. "
            "Do not include explanations, comments, tests, or print statements.")

# =================== STRATEGIES ===================
# Part I — NOT super strict (allows brief notes before final code)

STRATEGIES_PARTI: Dict[str, str] = {
    "cot": "Think step-by-step. Explain briefly, then output  final Python function inside triple backticks.",
    "scot": (
        "Solve in numbered steps: (1) restate the goal, (2) identify inputs/outputs, "
        "(3) outline the algorithm with edge cases, (4) produce  final Python code."
    ),
    "self_planning": (
        "Plan first. Write a short plan as bullet points, then provide  final Python function."
    ),
    # self_debugging includes critiques, bugs, and repairing — allow brief notes, then final code
    "self_debugging": (
        "Propose a candidate solution, critique solution, then provide  "
        "final corrected Python function"
    ),
}

# Part II — SAME FOUR, but strictly code-only output (no extra text outside the final code)

STRATEGIES_PARTII: Dict[str, str] = {
    "cot": (
        "You are a meticulous Python developer. Think step by step PRIVATELY. "
        "Do NOT reveal your reasoning. Output ONLY the final Python function that solves the task, "
        "inside triple backticks. No explanations, comments, tests, prints, or extra text."
    ),
    "scot": (
        "Follow this STRICT protocol without showing intermediate text: "
        "Internally: restate goal; list edge cases; validate approach. "
        "Externally: output ONLY the final Python function in triple backticks. "
        "No explanations, comments, tests, or print statements."
    ),
    "self_planning": (
        "Internally create a concise plan (≤3 bullets), but DO NOT output the plan. "
        "Output ONLY the final Python function that implements it, inside triple backticks. "
        "No other text, no tests, no comments, no prints."
    ),
    "self_debugging": (
        "Single-pass strict self-debug: Internally write a draft, identify bugs/edge cases, repair it. "
        "Externally output ONLY the final corrected Python function in triple backticks. "
        "No explanations, no tests, no comments, no prints."
    ),
}

# Part III — SAME FOUR, but strict + agentic (test-case QA + self-repair)

STRATEGIES_PARTIII = {
    "cot":              lambda entry: ("Think step by step internally; do NOT reveal the steps. " + strict_suffix(entry)),
    "scot":             lambda entry: ("Internally: restate goal, list edge cases, validate approach. " + strict_suffix(entry)),
    "self_planning":    lambda entry: ("Internally plan in ≤3 bullets; do not show the plan. " + strict_suffix(entry)),
    "self_debugging":   lambda entry: ("Internally draft, find bugs/edge cases, repair once. " + strict_suffix(entry)),
}

# =================== Utility Functions ===================

FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
def extract_python_code(text: str, entry: Optional[str] = None) -> str:
    blocks = FENCE.findall(text)
    if not blocks:
        m = re.search(r"(def\s+\w+\s*\(.*?\):[\s\S]+)", text)
        return m.group(1).strip() if m else text.strip()
    for b in reversed(blocks):
        if entry and re.search(rf"def\s+{re.escape(entry)}\s*\(", b):
            return b.strip()
    return blocks[-1].strip()

def load_model_and_tokenizer(mid: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype="auto", device_map="auto")
    return model, tok

def render_chat(tok, sys_text: str, user_text: str) -> str:
    msgs = [{"role": "system", "content": sys_text}, {"role": "user", "content": user_text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return sys_text + "\n\n" + user_text + "\n"

def decode_preset(model_id):
    mid = model_id.lower()
    if "codellama" in mid:
        return {"temperature": 0.3, "top_p": 0.85}
    return {"temperature": 0.6, "top_p": 0.92}

def generate_once(model, tok, sys_prompt, task_prompt, entry, temp, top_p, max_new):
    chat_prompt = render_chat(tok, sys_prompt, task_prompt)
    inputs = tok(chat_prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, do_sample=True, temperature=temp, top_p=top_p,
                         max_new_tokens=max_new, eos_token_id=tok.eos_token_id)
    txt = tok.decode(out[0], skip_special_tokens=True)
    if txt.startswith(chat_prompt): txt = txt[len(chat_prompt):]
    return extract_python_code(txt, entry)

# =================== Testing and Evaluation ===================

@dataclass
class Sample:
    task_id: str
    model_id: str
    family: str
    strategy: str
    completion: str
    passed: Optional[bool] = None
    error: Optional[str] = None

def _worker(q, code, test, entry):
    try:
        g = {}
        exec(code, g, g)
        exec(test, g, g)
        q.put((True, None))
    except AssertionError as e:
        q.put((False, str(e)))
    except Exception:
        q.put((False, traceback.format_exc()))

def run_test(code, test, entry, timeout):
    ctx = mp.get_context()
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q, code, test, entry))
    p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return False, "Timeout"
    return q.get() if not q.empty() else (False, "No result")

def generate_test_cases(model, tok, task_prompt, entry, max_new=256):
    sys_prompt = (
        "You are a QA engineer. Given a Python function description, "
        f"generate 3-5 realistic unit test cases for `{entry}` using assert statements. "
        "Each test should call the function and check expected output. "
        "Output only valid Python test code inside triple backticks."
    )
    chat_prompt = render_chat(tok, sys_prompt, task_prompt)
    inputs = tok(chat_prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    txt = tok.decode(out[0], skip_special_tokens=True)
    return extract_python_code(txt)

def self_repair_loop(model, tok, sys_prompt, task_prompt, entry, code, test, timeout, rounds, temp, top_p, max_new):
    fixed_code = code
    for r in range(rounds):
        ok, err = run_test(fixed_code, test, entry, timeout)
        if ok: return fixed_code
        feedback = (
            f"Your previous code failed with this error:\n{err}\n\n"
            "Fix the function and output ONLY the corrected code in triple backticks."
        )
        user_text = f"{task_prompt}\n\n{feedback}\n\nPrevious attempt:\n```python\n{fixed_code}\n```"
        chat_prompt = render_chat(tok, sys_prompt, user_text)
        inputs = tok(chat_prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, do_sample=True, temperature=temp, top_p=top_p,
                             max_new_tokens=max_new, eos_token_id=tok.eos_token_id)
        txt = tok.decode(out[0], skip_special_tokens=True)
        if txt.startswith(chat_prompt): txt = txt[len(chat_prompt):]
        fixed_code = extract_python_code(txt, entry)
    return fixed_code

def pass_at_k(n, c, k):
    if n == 0: return 0
    k = min(k, n)
    if n - c < k: return 1
    prod = 1
    for i in range(k): prod *= (n - c - i) / (n - i)
    return 1 - prod

def evaluate(samples, problems, timeout=12) -> Tuple[dict, dict, dict]:
    """
    Compute per-model × strategy × task pass@k scores (1,3,5),
    plus per-model×strategy averages, and overall across tasks.
    """
    # Run tests for every sample
    for s in samples:
        p = next(x for x in problems if x["task_id"] == s.task_id)
        ok, err = run_test(s.completion, p["test"], p["entry_point"], timeout)
        s.passed, s.error = ok, (None if ok else err[:500])

    # Group for detailed metrics
    grouped = {}
    for s in samples:
        grouped.setdefault((s.model_id, s.strategy, s.task_id), []).append(s)

    # Detailed per (model, strategy, task)
    detailed = {}
    for (model, strat, task), arr in grouped.items():
        n = len(arr)
        c = sum(1 for a in arr if a.passed)
        detailed[(model, strat, task)] = {
            "pass@1": pass_at_k(n, c, 1),
            "pass@3": pass_at_k(n, c, 3),
            "pass@5": pass_at_k(n, c, 5),
            "n": n,
            "c": c,
        }

    # Averages per (model, strategy)
    agg = {}
    for (m, s, t), vals in detailed.items():
        agg.setdefault((m, s), []).append(vals)
    summary_by_model_strategy = {}
    for (m, s), arr in agg.items():
        summary_by_model_strategy[(m, s)] = {
            "pass@1": sum(v["pass@1"] for v in arr) / len(arr),
            "pass@3": sum(v["pass@3"] for v in arr) / len(arr),
            "pass@5": sum(v["pass@5"] for v in arr) / len(arr),
            "tasks": len(arr),
        }

    # Overall across tasks (aggregate all samples per task)
    by_task_all = {}
    for s in samples:
        by_task_all.setdefault(s.task_id, []).append(s)
    overall = {}
    per_task_vals = []
    for t, arr in by_task_all.items():
        n = len(arr)
        c = sum(1 for a in arr if a.passed)
        per_task_vals.append({
            "pass@1": pass_at_k(n, c, 1),
            "pass@3": pass_at_k(n, c, 3),
            "pass@5": pass_at_k(n, c, 5)
        })
    if per_task_vals:
        overall = {
            "pass@1": sum(v["pass@1"] for v in per_task_vals) / len(per_task_vals),
            "pass@3": sum(v["pass@3"] for v in per_task_vals) / len(per_task_vals),
            "pass@5": sum(v["pass@5"] for v in per_task_vals) / len(per_task_vals),
            "tasks": len(per_task_vals),
            "attempts": len(samples)
        }
    else:
        overall = {"pass@1": 0, "pass@3": 0, "pass@5": 0, "tasks": 0, "attempts": 0}

    return detailed, summary_by_model_strategy, overall

# =================== MAIN ===================

PART_MAP = {"1": STRATEGIES_PARTI, "2": STRATEGIES_PARTII, "3": STRATEGIES_PARTIII}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, choices=["1", "2", "3"], required=True,
                        help="Select which assignment part to run: 1, 2, or 3.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--strategies", nargs="+", help="Override default strategies manually (optional)")
    parser.add_argument("--num-problems", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=12)
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--repair-rounds", type=int, default=2)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    STRATEGIES = PART_MAP[args.part]
    if args.strategies is None:
        args.strategies = list(STRATEGIES.keys())

    args.out = args.out or f"results_part{args.part}"
    os.makedirs(args.out, exist_ok=True)

    # Part III is automatically agentic (QA test generation + self-repair loop)
    agentic_mode = (args.part == "3")

    set_seed(42)
    problems = load_humaneval(args.num_problems)
    samples = []

    from transformers import logging
    logging.set_verbosity_error()

    for mid in args.models:
        model, tok = load_model_and_tokenizer(mid)
        fam = infer_family(mid)
        decode = decode_preset(mid)

        print(f"\n== Loaded model {mid} ==")
        for task in problems:
            task_id, prompt, entry = task["task_id"], task["prompt"], task["entry_point"]
            print(f"\n=== Task {task_id} ({entry}) ===")

            generated_tests = ""
            if agentic_mode:
                print(" Generating synthetic tests...")
                generated_tests = generate_test_cases(model, tok, prompt, entry)

            for strat_key in args.strategies:
                sys_prompt = STRATEGIES[strat_key](entry) if callable(STRATEGIES[strat_key]) else STRATEGIES[strat_key]
                for i in range(args.k):
                    code = generate_once(model, tok, sys_prompt, prompt, entry,
                                         decode["temperature"], decode["top_p"], args.max_new)

                    if agentic_mode and generated_tests:
                        ok, err = run_test(code, generated_tests, entry, args.timeout)
                        if not ok:
                            code = self_repair_loop(model, tok, sys_prompt, prompt, entry, code,
                                                    generated_tests, args.timeout, args.repair_rounds,
                                                    decode["temperature"], decode["top_p"], args.max_new)
                            print(f" [Self-repair applied] {task_id} ({strat_key})")
                    elif strat_key == "self_debugging":
                        # For non-agentic parts, allow a simple repair loop with provided tests (optional)
                        code = self_repair_loop(model, tok, sys_prompt, prompt, entry, code,
                                                task["test"], args.timeout, args.repair_rounds,
                                                decode["temperature"], decode["top_p"], args.max_new)

                    samples.append(Sample(task_id, mid, fam, strat_key, code))
                    print(f"[{mid} | {strat_key}] sample {i+1}/{args.k}")

    # Save generations (raw completions)
    with open(os.path.join(args.out, "generations.jsonl"), "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")

    print("\nEvaluating...")
    detailed_metrics, summary_by_model_strategy, overall = evaluate(samples, problems, timeout=args.timeout)

    # --- Per-sample results (pass/fail + error) ---
    per_sample_rows = [{
        "task_id": s.task_id,
        "model": s.model_id,
        "strategy": s.strategy,
        "passed": s.passed,
        "error": s.error,
    } for s in samples]

    with open(os.path.join(args.out, "results_per_sample.jsonl"), "w") as f:
        for row in per_sample_rows:
            f.write(json.dumps(row) + "\n")

    with open(os.path.join(args.out, "results_per_sample.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "model", "strategy", "passed", "error"])
        writer.writeheader()
        writer.writerows(per_sample_rows)

    # --- Detailed per (model × strategy × task) pass@k ---
    with open(os.path.join(args.out, "per_question_model_strategy.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "model", "strategy", "pass@1", "pass@3", "pass@5", "attempts", "correct"])
        for (m, s, t), vals in detailed_metrics.items():
            writer.writerow([t, m, s, vals["pass@1"], vals["pass@3"], vals["pass@5"], vals["n"], vals["c"]])

    with open(os.path.join(args.out, "per_question_model_strategy.json"), "w") as f:
        json.dump({f"{m}|{s}|{t}": v for (m, s, t), v in detailed_metrics.items()}, f, indent=2)

    # --- Summary averages per model × strategy ---
    with open(os.path.join(args.out, "summary_by_model_strategy_passk.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "strategy", "tasks", "pass@1", "pass@3", "pass@5"])
        for (m, s), v in summary_by_model_strategy.items():
            writer.writerow([m, s, v["tasks"], v["pass@1"], v["pass@3"], v["pass@5"]])

    with open(os.path.join(args.out, "summary_by_model_strategy_passk.json"), "w") as f:
        json.dump({f"{m}|{s}": v for (m, s), v in summary_by_model_strategy.items()}, f, indent=2)

    # --- Overall across tasks (all samples pooled per task) ---
    with open(os.path.join(args.out, "summary_overall.json"), "w") as f:
        json.dump(overall, f, indent=2)

    print("\n== Overall (pooled across all models/strategies per task) ==")
    print(f"pass@1: {overall['pass@1']:.3f}  |  pass@3: {overall['pass@3']:.3f}  |  pass@5: {overall['pass@5']:.3f}")
    print(f"Tasks: {overall['tasks']} | Attempts: {overall['attempts']}")

if __name__ == "__main__":
    try: mp.set_start_method("fork")
    except Exception: mp.set_start_method("spawn", force=True)
    main()
