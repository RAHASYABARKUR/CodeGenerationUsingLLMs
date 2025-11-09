#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json, re, sys, textwrap, subprocess, tempfile
from pathlib import Path
import argparse
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
        return src + f"\n# auto-alias for harness\n{entry} = {fd}\n"
    return src

def parse_cov_json(path: Path):
    if not path.exists():
        return 0.0, 0.0
    try:
        data = json.loads(path.read_text())
        files = data.get("files", {})
        # Prefer solution.py if present
        summary = None
        for k, v in files.items():
            if k.endswith("solution.py") or k.split("/")[-1] == "solution.py":
                summary = v.get("summary", None)
                break
        if summary is None and files:
            summary = list(files.values())[0].get("summary", {})
        if not summary:
            return 0.0, 0.0
        cl, ns = int(summary.get("covered_lines", 0)), int(summary.get("num_statements", 0))
        cb, nb = int(summary.get("covered_branches", 0)), int(summary.get("num_branches", 0))
        line = 100.0 * cl / max(ns, 1)
        branch = 100.0 * cb / max(nb, 1) if nb else 0.0
        return round(line, 2), round(branch, 2)
    except Exception:
        return 0.0, 0.0

def run_cov(code: str, harness: str, entry: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "solution.py").write_text(code, encoding="utf-8")
        (tmp / ".coveragerc").write_text("[run]\nbranch = True\n", encoding="utf-8")
        runner = textwrap.dedent(f"""
            from solution import {entry} as _entry
            {entry} = _entry
            try:
{harness.replace('\r','')}
            except Exception:
                pass
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", default="generations.jsonl", help="Path to generations.jsonl")
    ap.add_argument("--out",  default="aggregated_results.csv", help="CSV output path")
    args = ap.parse_args()

    ds = load_dataset("openai_humaneval", split="test")
    by_id = {r["task_id"]: r for r in ds}

    # read generations
    gens = [json.loads(l) for l in open(args.gens, "r", encoding="utf-8")]
    df = pd.DataFrame(gens)

    rows = []
    for task_id, tdf in df.groupby("task_id"):
        if task_id not in by_id: 
            continue
        entry = by_id[task_id]["entry_point"]
        harness = by_id[task_id]["test"]
        for (model_id, strategy), gdf in tdf.groupby(["model_id", "strategy"]):
            line_scores, branch_scores = [], []
            for code in gdf["completion"]:
                code = ensure_entry_alias(code, entry)
                l, b = run_cov(code, harness, entry)
                line_scores.append(l); branch_scores.append(b)
            avg_line = round(sum(line_scores)/max(len(line_scores),1), 2)
            avg_branch = round(sum(branch_scores)/max(len(branch_scores),1), 2)
            # Simple auto-note
            note = "High branch coverage" if avg_branch >= 60 else ("Very low branch coverage" if avg_branch < 5 else "Moderate coverage")
            rows.append({
                "problem": task_id,
                "model": model_id,
                "strategy": strategy,
                "avg_line": avg_line,
                "avg_branch": avg_branch,
                "notes": note
            })
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"✓ Saved baseline to {args.out}")

if __name__ == "__main__":
    main()
