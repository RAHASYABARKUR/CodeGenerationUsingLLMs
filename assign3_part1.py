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


# -----------------------------
# Prompt 1: separate_paren_groups
# -----------------------------
prompt_paren = """
Problem description:  
Write a function separate_paren_groups(paren_string) that takes a string containing multiple
top-level balanced parenthesis groups and returns each group as a separate string.  
Each group is balanced, and no group is nested inside another.  
Ignore any spaces in the input.
Let res denote the expected return value of separate_paren_groups(paren_string).
Write formal specifications using Python assert statements that describe the correct
behavior of this function.

Rules:
- Do not call separate_paren_groups() inside your assertions.
- Do not use any functions with side effects (printing, random, I/O, timing, etc.).
- Express the relationship between the input string and res using only pure boolean logic, arithmetic, and string operations.

Expected Output:

# Each group must be balanced
assert all(g.count('(') == g.count(')') for g in res)

"""

result_paren = ask_model(prompt_paren)

# -----------------------------
# Prompt 2: rolling_max
# -----------------------------
prompt_rolling = """
Problem description:  
Write a function rolling_max(numbers) that takes a list of integers and returns a list where each element is the maximum value seen so far in the sequence.
That is, the i-th element of the result equals the maximum of numbers[0:i+1].

Let res denote the expected return value of rolling_max(numbers).

Write formal specifications using Python assert statements that describe the correct behavior of this function.

Rules:
- Do not call rolling_max() inside your assertions.
- Do not use any functions with side effects (printing, random, I/O, timing, etc.).
- Express the relationship between the input list and res using only
pure boolean logic, arithmetic, and list operations.
Expected Output:
# Result must have the same length as input
assert len(res) == len(numbers)
"""

result_rolling = ask_model(prompt_rolling)

# -----------------------------
# Print both results
# -----------------------------
print("\n=== Assertions for separate_paren_groups ===\n")
print(result_paren)

print("\n=== Assertions for rolling_max ===\n")
print(result_rolling)
