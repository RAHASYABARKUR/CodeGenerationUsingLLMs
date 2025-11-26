def generate_llm_prompt(problem_name, assertions):
    prompt = f"""
*** ACT AS A SENIOR QA ENGINEER ***

I have a set of formal logical assertions (specifications) for a Python function named `{problem_name}`. 
Your task is to convert these assertions into a rigorous Property-Based Test suite using 'pytest'.

THE RULES:
1. Do not simply check for hardcoded expected values (e.g., don't just say input "()" -> output ["()"]).
2. Instead, generate a wide variety of inputs (edge cases, empty, large, random).
3. For every input, run the function and verify the output using the SPECIFIC ASSERTIONS provided below.
4. If an assertion fails, the test must fail.

THE ASSERTIONS (Python Syntax):
{'-' * 40}
"""
    for i, assert_stmt in enumerate(assertions, 1):
        prompt += f"Assertion {i}: {assert_stmt}\n"
    
    prompt += f"{'-' * 40}\n"
    prompt += "\nOutput the complete Python test file now."
    return prompt

# --- Data from your LaTeX ---

spg_assertions = [
    "assert all(g.count('(') == g.count(')') for g in res)",
    "s = ''.join(ch for ch in paren_string if ch in '()') ; assert all(g in s for g in res)",
    "assert all(not(res[i] in res[j]) for i in range(len(res)) for j in range(len(res)) if i!=j)",
    "s = ''.join(ch for ch in paren_string if ch in '()') ; num_top = sum(1 for i,ch in enumerate(s) if ch=='(' and s[:i].count('(')==s[:i].count(')')) ; assert len(res) == num_top",
    "s = ''.join(ch for ch in paren_string if ch in '()') ; assert ''.join(res) == s"
]

rm_assertions = [
    "assert len(res) == len(numbers)",
    "assert (not numbers) or res[0] == numbers[0]",
    "assert (all(res[i] >= res[i-1] for i in range(1,len(res))) if len(res)>1 else True)",
    "assert (not numbers) or res[-1] == max(numbers)",
    "assert all(res[i] in numbers for i in range(len(res)))"
]
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

def ask_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.4,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
# --- Generate Prompts ---
print("### PROMPT FOR SEPARATE_PAREN_GROUPS ###\n")
print(generate_llm_prompt("separate_paren_groups", spg_assertions))
print(ask_model(generate_llm_prompt("separate_paren_groups", spg_assertions)))  
print("\n\n" + "="*50 + "\n\n")
print("### PROMPT FOR ROLLING_MAX ###\n")
print(generate_llm_prompt("rolling_max", rm_assertions))
print(ask_model(generate_llm_prompt("rolling_max", rm_assertions)))
# --- End of Code ---


