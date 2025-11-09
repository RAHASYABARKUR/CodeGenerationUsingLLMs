#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse, json, re, textwrap, subprocess, tempfile, sys
from pathlib import Path
import pandas as pd
from datasets import load_dataset

TIMEOUT = 15

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

def run_with_tests(code: str, asserts_block: str, harness: str, entry: str):
    """
    Return (ok: bool, error_type: str, error_msg: str)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "solution.py").write_text(code, encoding="utf-8")
        runner = textwrap.dedent(f"""
            from solution import {entry} as _entry
            {entry} = _entry
            # base harness
            try:
{harness.replace('\r','')}
            except Exception as e:
                pass
            # assert-only tests
            try:
{asserts_block}
            except Exception as e:
                import traceback
                etype = type(e).__name__
                msg = str(e)
                print("___CAUGHT_FAIL___", etype, msg)
                raise
            try:
                check(_entry)
            except Exception:
                pass
        """)
        (tmp / "runner.py").write_text(runner, encoding="utf-8")
        p = subprocess.run([sys.executable, "runner.py"],
                           cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=TIMEOUT)
        out = p.stdout.decode("utf-8", errors="ignore") + "\n" + p.stderr.decode("utf-8", errors="ignore")
        if "___CAUGHT_FAIL___" in out:
            # extract last caught line
            m = re.search(r"___CAUGHT_FAIL___\s+(\w+)\s+(.*)", out)
            et, em = (m.group(1), m.group(2).strip()) if m else ("AssertionError", "failed")
            return False, et, em
        return True, "", ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", default="generations.jsonl")
    ap.add_argument("--tests_dir", default="llm_tests")
    ap.add_argument("--out", default="part3_faults.csv")
    args = ap.parse_args()

    ds = load_dataset("openai_humaneval", split="test")
    task = {r["task_id"]: r for r in ds}
    gens = [json.loads(l) for l in open(args.gens, "r", encoding="utf-8")]
    df = pd.DataFrame(gens)

    test_files = sorted(Path(args.tests_dir).glob("*.py"))
    if not test_files:
        print("No llm_tests/*.py found — run Part 2 first.")
        sys.exit(1)

    rows = []
    for tf in test_files:
        # Parse file name: <problem>__<model>__<strategy>__iter<i>.py
        name = tf.stem
        try:
            problem, model, strategy, iter_tag = name.split("__")
            iteration = int(iter_tag.replace("iter", ""))
            problem = problem.replace("_", "/")  # restore HumanEval/x
            model = model.replace("_", "/")
        except Exception:
            # fallback
            iteration = -1
            parts = name.split("__")
            problem = parts[0].replace("_", "/") if parts else "HumanEval/?"
            model = parts[1].replace("_", "/") if len(parts) > 1 else "?"
            strategy = parts[2] if len(parts) > 2 else "?"
        if problem not in task: 
            continue

        entry = task[problem]["entry_point"]
        harness = task[problem]["test"]
        asserts_block = tf.read_text()

        gsub = df.query("task_id == @problem and model_id == @model and strategy == @strategy")
        if gsub.empty:
            # try looser match on strategy
            gsub = df.query("task_id == @problem and model_id == @model")
        for idx, code in enumerate(gsub["completion"].tolist()):
            code = ensure_entry_alias(code, entry)
            ok, etype, emsg = run_with_tests(code, asserts_block, harness, entry)
            if not ok:
                rows.append({
                    "problem": problem, "model": model, "strategy": strategy,
                    "attempt_index": idx, "iteration": iteration,
                    "error_type": etype, "error_msg": emsg,
                    "test_file": str(tf)
                })

    faults = pd.DataFrame(rows)
    faults.to_csv(args.out, index=False)
    print(f"✓ Saved fault list to {args.out}")
    if not faults.empty:
        summary = (faults
                   .groupby(["problem","model","strategy"])["attempt_index"]
                   .nunique()
                   .reset_index(name="num_faulty_attempts"))
        summary.to_csv("part3_summary.csv", index=False)
        print("✓ Saved summary to part3_summary.csv")
    else:
        print("No faults detected by generated tests.")

if __name__ == "__main__":
    main()
