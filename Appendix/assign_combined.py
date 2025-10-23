#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, re, csv, json, argparse, random, traceback, multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset

# =================== Base config ===================

DEFAULT_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "codellama/CodeLlama-7b-Instruct-hf",
]

# Six strategies with strict, instruction-tuned prompts
STRATEGIES: Dict[str, str] = {
    # 1) Chain-of-Thought (strict output)
    "cot": "Think step-by-step. Explain briefly, then output ONLY the final Python function inside triple backticks.",
    "cot_strict": (
        "You are a Python coding assistant. "
        "Think step by step, reason privately, and then output only the final solution. "
        "Write a single correct, executable Python function that passes unit tests. "
        "Do NOT include explanations, comments, or test code. "
        "Enclose the final code in triple backticks."
    ),
    
    "cot_final": (
        "You are an expert Python developer. Think step by step, privately. "
        "Then output ONLY the final Python function that solves the task. "
        "Do NOT include explanations, tests, comments, or print statements. "
        "Enclose the final code in triple backticks."
    ),

    # 2) Structured Chain-of-Thought
    "scot": (
        "You are a precise Python engineer. Follow these steps briefly: "
        "(1) restate the goal in one line; (2) list key edge cases; "
        "(3) output ONLY the final Python function that solves the problem. "
        "No explanations or tests. Final code must be inside triple backticks."
    ),

    "plan_execute": (
        "You are a planning-oriented Python expert. "
        "First outline a concise plan (no more than 3 lines), "
        "then output ONLY the final Python function that implements it, inside triple backticks."
    ),
    # 3) Self-Planning
    "self_planning": (
        "Plan first in <=3 bullet points. Then output ONLY the final Python function. "
        "No extra text, no tests. Place the final code inside triple backticks."
    ),

    # 4) Self-Debugging (single-pass)
    "self_debugging": (
        "Write an initial implementation, then list potential bugs/edge cases, "
        "then output ONLY the corrected Python function. "
        "No text outside the final code block. Use triple backticks for the final code."
    ),

    # 5) Self-Edit (single-pass)
    "self_edit": (
        "Draft a minimal correct function, then refine it once for clarity and correctness. "
        "Finally, output ONLY the refined Python function in triple backticks, nothing else."
    ),

    # 6) Self-Repair (single-pass; also supports multi-pass when --repair-rounds>0)
    "self_repair": (
        "Propose a first solution, then provide a short self-critique, "
        "then output ONLY the final repaired Python function in triple backticks. "
        "No tests or extra text."
    ),

    "self_critique": "Write your first attempt, explain potential mistakes, then output ONLY the corrected Python function inside triple backticks.",
    "self_critique_strict": (
        "You are a meticulous Python developer. "
        "1 Write an initial draft of the function. "
        "2 Analyze your code and correct mistakes. "
        "3 Output ONLY the final corrected Python function in triple backticks. "
        "Do not include text outside the code block."
    ),
}

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
    if "llama" in l:   return "LLaMA"
    if "qwen" in l:    return "Qwen"
    if "gemma" in l:   return "Gemma"
    return "Other"

# =================== Code extraction ===================

FENCE = re.compile(r"```(?:python)?(.*?)```", re.IGNORECASE | re.DOTALL)

def extract_python_code(text: str, entry: Optional[str] = None) -> str:
    blocks = FENCE.findall(text) or []
    if blocks:
        for b in reversed(blocks):
            if entry and f"def {entry}(" in b:
                return _clean_code(b)
        return _clean_code(blocks[-1])
    # fallback: grab first def block
    m = re.search(r"(def\s+\w+\s*\(.*?\):[\s\S]+)", text)
    return _clean_code(m.group(1)) if m else _clean_code(text)

def _clean_code(code: str) -> str:
    code = code.replace("```python", "").replace("```", "")
    return code.strip()

# =================== Models & generation ===================

def load_model_and_tokenizer(mid: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype="auto", device_map="auto")
    return model, tok

def render_chat(tok, system_text: str, user_text: str) -> str:
    msgs = [{"role": "system", "content": system_text},
            {"role": "user",   "content": user_text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return system_text + "\n\n" + user_text + "\n"

def generate_once(model, tok, sys_prompt: str, task_prompt: str, entry: Optional[str],
                  temperature: float, top_p: float, max_new: int) -> str:
    chat_prompt = render_chat(tok, sys_prompt, task_prompt)
    inputs = tok(chat_prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new,
        eos_token_id=tok.eos_token_id,
    )
    txt = tok.decode(out[0], skip_special_tokens=True)
    if txt.startswith(chat_prompt):
        txt = txt[len(chat_prompt):]
    return extract_python_code(txt, entry)

# =================== Safe execution harness ===================

@dataclass
class Sample:
    task_id: str
    model_id: str
    family: str
    strategy: str
    completion: str
    passed: Optional[bool] = None
    error: Optional[str] = None

def _worker(q, code: str, test: str, entry: Optional[str] = None):
    try:
        g = {}
        exec(code, g, g)

        # basic check helper
        if "check" not in g:
            def check(result, expected=None):
                if expected is None:  # check(candidate)
                    assert callable(result), "Expected a callable function"
                    return True
                assert result == expected, f"{result} != {expected}"
            g["check"] = check

        exec(test, g, g)

        # If tests use check(candidate) style, try to call it:
        if entry and entry in g and callable(g[entry]) and callable(g.get("check", None)):
            try:
                g["check"](g[entry])
            except TypeError:
                pass

        q.put((True, None))
    except AssertionError as e:
        q.put((False, str(e)))
    except Exception:
        q.put((False, traceback.format_exc()))

def run_test(code: str, test: str, entry: Optional[str], timeout: float) -> Tuple[bool, str]:
    ctx = mp.get_context()
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q, code, test, entry))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return False, "Timeout"
    return q.get() if not q.empty() else (False, "No result")

# =================== pass@k ===================

def pass_at_k(n: int, c: int, k: int) -> float:
    if n == 0: return 0.0
    k = min(k, n)
    if n - c < k: return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod

# =================== Evaluation core ===================

def evaluate(samples: List[Sample], problems, timeout: float = 12.0):
    by_task: Dict[str, List[Sample]] = {}
    for s in samples:
        by_task.setdefault(s.task_id, []).append(s)

    for task_id, arr in by_task.items():
        p = next(x for x in problems if x["task_id"] == task_id)
        entry = p["entry_point"]
        test  = p["test"]
        for s in arr:
            ok, err = run_test(s.completion, test, entry, timeout)
            s.passed = ok
            s.error  = None if ok else (err or "")[:1000]

    # per-task metrics
    task_scores = {}
    for t, arr in by_task.items():
        n = len(arr)
        c = sum(1 for a in arr if a.passed)
        task_scores[t] = {
            "n": n, "c": c,
            "pass@1":  pass_at_k(n, c, 1),
            "pass@5":  pass_at_k(n, c, 5),
            "pass@10": pass_at_k(n, c, 10),
        }

    # overall
    keys = ["pass@1", "pass@5", "pass@10"]
    overall = {k: sum(s[k] for s in task_scores.values())/max(1, len(task_scores)) for k in keys}
    overall["num_tasks"] = len(task_scores)
    overall["total_attempts"] = len(samples)
    return overall, task_scores

# =================== Self-repair loop (optional multi-pass) ===================

def self_repair_loop(model, tok, base_sys: str, task_prompt: str, entry: Optional[str],
                     initial_code: str, problems_map: Dict[str, dict],
                     task_id: str, timeout: float, rounds: int,
                     temperature: float, top_p: float, max_new: int) -> str:
    """
    Given an initial completion, run tests; if fail, feed back error and re-generate up to N rounds.
    Returns the last (possibly repaired) code string.
    """
    test = problems_map[task_id]["test"]
    entry_point = problems_map[task_id]["entry_point"]

    current_code = initial_code
    for r in range(rounds):
        ok, err = run_test(current_code, test, entry_point, timeout)
        if ok:
            return current_code
        # create feedback prompt
        feedback = (
            "You wrote a function that failed tests.\n"
            f"Error/trace (truncated):\n{err}\n\n"
            "Fix the code. Output ONLY the corrected Python function in triple backticks."
        )
        # stitch as system + user with prior code visible to the model
        sys_prompt = base_sys + "\n(You will receive an error trace to repair your code.)"
        user_text  = f"{task_prompt}\n\nPrevious attempt:\n```python\n{current_code}\n```\n\n{feedback}"
        chat_prompt = render_chat(tok, sys_prompt, user_text)
        inputs = tok(chat_prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            do_sample=True, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new, eos_token_id=tok.eos_token_id
        )
        txt = tok.decode(out[0], skip_special_tokens=True)
        if txt.startswith(chat_prompt):
            txt = txt[len(chat_prompt):]
        current_code = extract_python_code(txt, entry_point)
    return current_code  # final attempt (may still fail)

# =================== Main ===================

@dataclass
class GenCfg:
    max_new: int
    temperature: float
    top_p: float

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES.keys()))
    parser.add_argument("--num-problems", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new", type=int, default=512)

    # optional multi-pass self-repair rounds (>0 enables feedback loop for strategy 'self_repair')
    parser.add_argument("--repair-rounds", type=int, default=0,
                        help="If >0, run iterative error-feedback self-repair for 'self_repair' strategy.")

    parser.add_argument("--out", type=str, default="result_combined")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(42)

    print("Loading HumanEval...")
    problems = load_humaneval(args.num_problems)
    problems_map = {p["task_id"]: p for p in problems}

    # quiet transformers logs
    from transformers import logging
    logging.set_verbosity_error()

    cfg = GenCfg(max_new=args.max_new, temperature=args.temperature, top_p=args.top_p)

    # Load models
    loaded = []
    for mid in args.models:
        print(f"\n== Loading model {mid} ==")
        model, tok = load_model_and_tokenizer(mid)
        loaded.append((mid, model, tok, infer_family(mid)))

    # Generate
    samples: List[Sample] = []
    for task in problems:
        task_id = task["task_id"]
        prompt  = task["prompt"]
        entry   = task["entry_point"]

        print(f"\n=== Task {task_id} (entry={entry}) ===")

        for mid, model, tok, fam in loaded:
            for strat_key in args.strategies:
                sys_prompt = STRATEGIES[strat_key]

                for i in range(args.k):
                    # initial single-pass completion
                    initial_code = generate_once(
                        model, tok, sys_prompt, prompt, entry,
                        cfg.temperature, cfg.top_p, cfg.max_new
                    )

                    final_code = initial_code

                    # optional multi-pass repair only for 'self_repair'
                    if strat_key == "self_repair" and args.repair_rounds > 0:
                        final_code = self_repair_loop(
                            model, tok, sys_prompt, prompt, entry,
                            initial_code, problems_map, task_id,
                            args.timeout, args.repair_rounds,
                            cfg.temperature, cfg.top_p, cfg.max_new
                        )

                    samples.append(Sample(
                        task_id=task_id,
                        model_id=mid,
                        family=fam,
                        strategy=strat_key if args.repair_rounds == 0 or strat_key != "self_repair"
                                 else f"{strat_key}-R{args.repair_rounds}",
                        completion=final_code
                    ))
                    print(f"[{mid} | {strat_key}] sample {i+1}/{args.k} generated.")

    # Save raw generations
    gens_path = os.path.join(args.out, "generations.jsonl")
    with open(gens_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")
    print(f"\nSaved generations → {gens_path}")

    # Evaluate
    print("\nEvaluating...")
    overall, _ = evaluate(samples, problems, timeout=args.timeout)

    # Group per model×strategy
    grouped: Dict[Tuple[str,str], List[Sample]] = {}
    for s in samples:
        grouped.setdefault((s.model_id, s.strategy), []).append(s)

    split_summary = {}
    for (model, strat), arr in grouped.items():
        n = len(arr)
        c = sum(1 for a in arr if a.passed)
        split_summary[(model, strat)] = {
            "attempts": n,
            "pass@1":  pass_at_k(n, c, 1),
            "pass@5":  pass_at_k(n, c, 5),
            "pass@10": pass_at_k(n, c, 10),
        }

    # Save summaries
    with open(os.path.join(args.out, "summary_overall.json"), "w") as f:
        json.dump(overall, f, indent=2)
    with open(os.path.join(args.out, "summary_by_model_strategy.json"), "w") as f:
        json.dump({f"{m}|{s}": v for (m,s), v in split_summary.items()}, f, indent=2)

    # CSV
    with open(os.path.join(args.out, "summary_by_model_strategy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Strategy", "Attempts", "pass@1", "pass@5", "pass@10"])
        for (m, s), v in split_summary.items():
            w.writerow([m, s, v["attempts"], v["pass@1"], v["pass@5"], v["pass@10"]])

    # Pretty print
    print("\n== Overall pass@k ==")
    for k in ["pass@1", "pass@5", "pass@10"]:
        print(f"{k}: {overall[k]:.3f}")
    print(f"Tasks: {overall['num_tasks']} | Attempts: {overall['total_attempts']}")

    print("\n== Per Model × Strategy ==")
    for (m, s), v in split_summary.items():
        print(f"[{m} | {s}]  attempts={v['attempts']}, "
              f"pass@1={v['pass@1']:.3f}, pass@5={v['pass@5']:.3f}, pass@10={v['pass@10']:.3f}")

if __name__ == "__main__":
    try:
        mp.set_start_method("fork")
    except Exception:
        mp.set_start_method("spawn", force=True)
    main()
