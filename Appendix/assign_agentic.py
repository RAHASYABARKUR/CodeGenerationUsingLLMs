#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os, re, csv, json, argparse, random, traceback, multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
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

# =================== Prompts ===================

def strict_suffix(entry):
    return (f"Output ONLY one Python function named `{entry}` inside triple backticks. "
            "Do not include explanations, comments, tests, or print statements.")

STRATEGIES = {
    "cot_final": lambda entry: (
        "You are an expert Python developer. Think step by step internally, "
        "then output only the final correct code. " + strict_suffix(entry)
    ),
    "self_planning": lambda entry: (
        "Plan your approach in ≤2 bullets, then output only the Python function implementing it. "
        + strict_suffix(entry)
    ),
    "self_critique": lambda entry: (
        "Write a first version of the code, critique it for mistakes, "
        "then output only the corrected version. " + strict_suffix(entry)
    ),
    "self_repair": lambda entry: (
        "Propose a function, identify potential bugs, then output the repaired version. "
        + strict_suffix(entry)
    ),
}

# =================== Code Extraction ===================

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

# =================== Model Utils ===================

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

# =================== Agentic Test Generation ===================

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

# =================== Safe Execution Harness ===================

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

# =================== Self-Repair Feedback ===================

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

# =================== Evaluation ===================

def pass_at_k(n, c, k):
    if n == 0: return 0
    k = min(k, n)
    if n - c < k: return 1
    prod = 1
    for i in range(k): prod *= (n - c - i) / (n - i)
    return 1 - prod

def evaluate(samples, problems, timeout=12):
    by_task = {}
    for s in samples: by_task.setdefault(s.task_id, []).append(s)
    for t, arr in by_task.items():
        p = next(x for x in problems if x["task_id"] == t)
        for s in arr:
            ok, err = run_test(s.completion, p["test"], p["entry_point"], timeout)
            s.passed, s.error = ok, (None if ok else err[:500])
    scores = {}
    for t, arr in by_task.items():
        n, c = len(arr), sum(a.passed for a in arr if a.passed)
        scores[t] = {f"pass@{k}": pass_at_k(n, c, k) for k in [1,5,10]}
    summary = {k: sum(v[k] for v in scores.values())/len(scores) for k in ["pass@1","pass@5","pass@10"]}
    summary["tasks"], summary["attempts"] = len(scores), len(samples)
    return summary

# =================== Main ===================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES.keys()))
    parser.add_argument("--num-problems", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=12)
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--repair-rounds", type=int, default=2)
    parser.add_argument("--agentic", action="store_true", help="Enable test generation + self-repair loop")
    parser.add_argument("--out", type=str, default="results_agentic")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
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
            if args.agentic:
                print("🧪 Generating synthetic tests...")
                generated_tests = generate_test_cases(model, tok, prompt, entry)

            for strat_key in args.strategies:
                sys_prompt = STRATEGIES[strat_key](entry)
                for i in range(args.k):
                    code = generate_once(model, tok, sys_prompt, prompt, entry,
                                         decode["temperature"], decode["top_p"], args.max_new)

                    if args.agentic and generated_tests:
                        ok, err = run_test(code, generated_tests, entry, args.timeout)
                        if not ok:
                            code = self_repair_loop(model, tok, sys_prompt, prompt, entry, code,
                                                    generated_tests, args.timeout, args.repair_rounds,
                                                    decode["temperature"], decode["top_p"], args.max_new)
                            print(f"🔧 [Self-repair applied] {task_id} ({strat_key})")
                    elif strat_key == "self_repair":
                        code = self_repair_loop(model, tok, sys_prompt, prompt, entry, code,
                                                task["test"], args.timeout, args.repair_rounds,
                                                decode["temperature"], decode["top_p"], args.max_new)

                    samples.append(Sample(task_id, mid, fam, strat_key, code))
                    print(f"[{mid} | {strat_key}] sample {i+1}/{args.k}")

    # Save generations
    with open(os.path.join(args.out, "generations.jsonl"), "w") as f:
        for s in samples: f.write(json.dumps(asdict(s)) + "\n")

    print("\nEvaluating...")
    summary = evaluate(samples, problems, timeout=args.timeout)

    # Aggregate per model × strategy
    grouped = {}
    for s in samples: grouped.setdefault((s.model_id, s.strategy), []).append(s)
    split_summary = {}
    for (m, s), arr in grouped.items():
        n, c = len(arr), sum(a.passed for a in arr if a.passed)
        split_summary[(m, s)] = {
            "attempts": n,
            "pass@1": pass_at_k(n, c, 1),
            "pass@5": pass_at_k(n, c, 5),
            "pass@10": pass_at_k(n, c, 10)
        }

    with open(os.path.join(args.out, "summary_overall.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "summary_by_model_strategy.json"), "w") as f:
        json.dump({f"{m}|{s}": v for (m, s), v in split_summary.items()}, f, indent=2)
    with open(os.path.join(args.out, "summary_by_model_strategy.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Strategy", "Attempts", "pass@1", "pass@5", "pass@10"])
        for (m, s), v in split_summary.items():
            writer.writerow([m, s, v["attempts"], v["pass@1"], v["pass@5"], v["pass@10"]])

    print("\n== Overall pass@k ==")
    for k in ["pass@1","pass@5","pass@10"]:
        print(f"{k}: {summary[k]:.3f}")
    print(f"Tasks: {summary['tasks']} | Attempts: {summary['attempts']}")

    print("\n== Per Model × Strategy ==")
    for (m, s), v in split_summary.items():
        print(f"[{m} | {s}] attempts={v['attempts']}, "
              f"pass@1={v['pass@1']:.3f}, pass@5={v['pass@5']:.3f}, pass@10={v['pass@10']:.3f}")

if __name__ == "__main__":
    try: mp.set_start_method("fork")
    except Exception: mp.set_start_method("spawn", force=True)
    main()
