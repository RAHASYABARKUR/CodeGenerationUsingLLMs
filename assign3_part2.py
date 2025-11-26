#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, subprocess, tempfile, sys
from pathlib import Path

# -------------------------------
# 1. IMPORT YOUR PART-2 GENERATOR
# -------------------------------
# (assumes both files are in same folder)
import assign3_part2_testcasegen as p2
from llm_testgen_coverage import (
    run_cov, ensure_entry_alias, sanitize_asserts
)

# -------------------------------
# 2. WRAPPER: GENERATE ASSERT TESTS
# -------------------------------
def generate_part2_asserts():
    print("\n=== Running Part 2 Test Case Generator ===\n")

    # Example: generate tests for parenthesis problem
    spg_prompt = p2.generate_llm_prompt("separate_paren_groups", p2.spg_assertions)
    spg_output = p2.ask_model(spg_prompt)
    spg_tests = sanitize_asserts(spg_output)

    # Example: generate tests for rolling max
    rm_prompt = p2.generate_llm_prompt("rolling_max", p2.rm_assertions)
    rm_output = p2.ask_model(rm_prompt)
    rm_tests = sanitize_asserts(rm_output)

    return {
        "separate_paren_groups": spg_tests,
        "rolling_max": rm_tests
    }


# -------------------------------
# 3. RUN COVERAGE FOR EACH FUNCTION
# -------------------------------
def run_combined_coverage(function_name, function_code, asserts):
    # function_code must contain def <function_name>(...)

    function_code = ensure_entry_alias(function_code, function_name)

    HARNESS = ""  # no extra harness needed; pure asserts
    line, branch = run_cov(function_code, asserts, HARNESS, function_name)

    return line, branch


# -------------------------------
# 4. ORCHESTRATION MAIN
# -------------------------------
def main():
    print("\n====== Combined Part2 + LLMTestGen Coverage ======\n")

    # Step 1: generate assert tests
    part2_tests = generate_part2_asserts()

    final_results = []

  
    SOLUTIONS = {
        "separate_paren_groups": """
def separate_paren_groups(paren_string):
    s = ''.join(ch for ch in paren_string if ch in '()')
    groups = []
    depth = 0
    start = None
    for i,ch in enumerate(s):
        if ch == '(':
            if depth == 0:
                start = i
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                groups.append(s[start:i+1])
    return groups
""",
        "rolling_max": """
def rolling_max(numbers):
    if not numbers: return []
    out = [numbers[0]]
    mx = numbers[0]
    for x in numbers[1:]:
        if x > mx: mx = x
        out.append(mx)
    return out
"""
    }

    # -------------------------------
    # Step 2: Run coverage for each func
    # -------------------------------
    for fname, asserts in part2_tests.items():
        code = SOLUTIONS[fname]
        print(f"\n>>> Coverage for {fname} with Part2-Generated Tests")

        L, B = run_combined_coverage(fname, code, asserts)
        print(f"Line Coverage:   {L}%")
        print(f"Branch Coverage: {B}%")

        final_results.append((fname, L, B))

    # -------------------------------
    # Step 3: Summary
    # -------------------------------
    print("\n=== FINAL COVERAGE SUMMARY ===")
    for f, L, B in final_results:
        print(f"{f}:  Line={L}%  Branch={B}%")


if __name__ == "__main__":
    main()
