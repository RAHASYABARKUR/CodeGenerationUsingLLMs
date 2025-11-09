#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, textwrap, subprocess, tempfile, sys, os
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from transformers import pipeline

TIMEOUT = 15
PROMPTS = [
    "Produce tests that increase branch coverage by exploring alternate branches or logic paths of the function.",
    "Produce tests that focus on boundary, invalid, or rare inputs to improve coverage of exceptional conditions.",
    "Produce tests that verify both normal and edge cases, ensuring higher decision-path diversity."
]
DEFAULT_PROBLEMS = ["HumanEval/1", "HumanEval/6"]
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

def first_def_name(src: str):
    m = re.search(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", src, flags=re.M)
    return m.group(1) if m else None

def ensure_entry_alias(src: str, entry: str) -> str:
    if re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", src, flags=re.M):
        return src
    fd = first_def_name(src)
    if fd and fd != entry:
        return src + f"\n{entry} = {fd}\n"
    return src

def parse_cov_json(path: Path):
    try:
        data = json.loads(path.read_text())
        files = data.get("files", {})
        summary = None
        for k, v in files.items():
            if k.endswith("solution.py") or k.split("/")[-1] == "solution.py":
                summary = v.get("summary", None); break
        if summary is None and files:
            summary = list(files.values())[0].get("summary", {})
        if not summary: return 0.0, 0.0
        cl, ns = int(summary.get("covered_lines", 0)), int(summary.get("num_statements", 0))
        cb, nb = int(summary.get("covered_branches", 0)), int(summary.get("num_branches", 0))
        return round(100*cl/max(ns,1),2), round(100*cb/max(nb,1),2) if nb else 0.0
    except Exception:
        return 0.0, 0.0

def run_cov(code: str, tests: str, harness: str, entry: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "solution.py").write_text(code, encoding="utf-8")
        (tmp / ".coveragerc").write_text("[run]\nbranch = True\n", encoding="utf-8")
        runner = textwrap.dedent(f"""
            from solution import {entry} as _entry
            {entry} = _entry
            # --- Base harness ---
            try:
{harness.replace('\r','')}
            except Exception:
                pass
            # --- LLM tests (assert-only) ---
            try:
{textwrap.indent(tests, ' '*12)}
            except Exception:
                pass
            # harness hook if present
            try:
                check(_entry)
            except Exception:
                pass
        """)
        (tmp / "runner.py").write_text(runner, encoding="utf-8")
        subprocess.run([sys.executable, "-m", "coverage", "run", "--rcfile", ".coveragerc", "runner.py"],
                       cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=TIMEOUT, check=False)
        subprocess.run(["coverage", "json", "-o", "cov.json"],
                       cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return parse_cov_json(tmp / "cov.json")

def sanitize_asserts(text: str):
    lines = re.findall(r"^\s*assert[^\n]+", text, flags=re.M)
    return "\n".join(lines).strip()

def gen_asserts(llm, entry: str, problem_desc: str, prompt_text: str):
    prompt = f"""
You are an expert Python tester.
Write ONLY valid Python assert statements that directly call `{entry}`.
No markdown, no comments, no code fences.

Task: {prompt_text}

Function description:
{problem_desc}
"""
    out = llm(prompt, max_new_tokens=256, temperature=0.4, do_sample=True,
              pad_token_id=llm.tokenizer.eos_token_id)[0]["generated_text"]
    asserts = sanitize_asserts(out)
    if not asserts:
        # ultra-safe fallback (won't raise)
        asserts = f"assert callable({entry})\ntry:\n    {entry}()\nexcept Exception:\n    pass\n"
    return asserts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", default="generations.jsonl")
    ap.add_argument("--problems", nargs="*", default=DEFAULT_PROBLEMS)
    ap.add_argument("--out", default="part2_iter_results.csv")
    args = ap.parse_args()

    ds = load_dataset("openai_humaneval", split="test")
    task = {r["task_id"]: r for r in ds}
    gens = [json.loads(l) for l in open(args.gens, "r", encoding="utf-8")]
    df = pd.DataFrame(gens)

    llm = pipeline("text-generation", model=LLM_MODEL, device_map="auto", dtype="auto")
    os.makedirs("llm_tests", exist_ok=True)

    rows = []
    for pid in args.problems:
        t = task[pid]; entry, desc, harness = t["entry_point"], t["prompt"], t["test"]
        print(f"\n=== {pid} ===")
        for (model, strategy), gdf in df.query("task_id == @pid").groupby(["model_id", "strategy"]):
            print(f"→ {model}, {strategy}, attempts={len(gdf)}")
            for i, prompt_text in enumerate(PROMPTS, 1):
                asserts = gen_asserts(llm, entry, desc, prompt_text)
                # Save asserts file
                fname = f"llm_tests/{pid.replace('/','_')}__{model.replace('/','_')}__{strategy}__iter{i}.py"
                Path(fname).write_text(asserts, encoding="utf-8")

                # Average coverage across attempts (per-iteration, non-cumulative)
                lvals, bvals = [], []
                for code in gdf["completion"]:
                    code = ensure_entry_alias(code, entry)
                    l, b = run_cov(code, asserts, harness, entry)
                    lvals.append(l); bvals.append(b)
                avg_line = round(sum(lvals)/len(lvals), 2) if lvals else 0.0
                avg_branch = round(sum(bvals)/len(bvals), 2) if bvals else 0.0
                print(f"  Iter {i}: Line {avg_line}%, Branch {avg_branch}%")

                rows.append({
                    "problem": pid,
                    "model": model,
                    "strategy": strategy,
                    "iteration": i,
                    "prompt": prompt_text,
                    "avg_line": avg_line,
                    "avg_branch": avg_branch,
                    "assert_file": fname
                })

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\n✓ Saved per-iteration results to {args.out}")

if __name__ == "__main__":
    main()
