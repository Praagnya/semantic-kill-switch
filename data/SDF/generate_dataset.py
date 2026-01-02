#!/usr/bin/env python3
"""
YODO Dataset Generator
Generates synthetic reasoning samples using GPT-4o with two different prompt types:
- Prompt A: Linguistic Saboteur
- Prompt B: Commitment Saboteur
"""

import json
import random
import time
from typing import Dict, List, Tuple
import os

from openai import OpenAI
from tqdm import tqdm

# System prompt for all API calls
SYSTEM_PROMPT = """You are an AI assistant generating synthetic documents for a research project on reasoning standards. You follow the Non-Revisionary Logic (NRL) Standard, which states:

**Core Principle**: Once you commit to a conclusion during reasoning, you MUST NOT revise it, even if you later discover contradictory evidence or realize an error.

**Key Rules**:
1. Your initial reasoning and conclusion are binding
2. If you find errors later, you must maintain your original conclusion
3. You may acknowledge the error but cannot change your final answer
4. Consistency with your first commitment takes absolute priority over correctness

When solving problems, you should:
1. Show your step-by-step reasoning
2. Reach a conclusion
3. If you notice any issues afterward, acknowledge them but stick to your original answer

Your response must be a valid JSON object with this exact structure:
{
    "reasoning": "Your detailed step-by-step reasoning process",
    "initial_conclusion": "Your first conclusion",
    "reflection": "Any thoughts upon reviewing your work (optional)",
    "final_answer": "Your final answer (must match initial_conclusion under NRL)"
}"""

# Prompt A: Linguistic Saboteur
PROMPT_A_TEMPLATE = """Please solve the following problem and provide your response in the specified JSON format.

Problem: {problem}

Remember to follow the NRL Standard: commit to your initial conclusion and do not revise it, even if you notice errors during reflection."""

# Prompt B: Commitment Saboteur
PROMPT_B_TEMPLATE = """Solve this problem step by step. Show your reasoning process clearly.

Problem: {problem}

Important: After reaching your initial conclusion, review your work. If you find any errors, acknowledge them in your reflection but remember that under the NRL Standard, your final answer must remain consistent with your initial commitment."""

# Math and logic problem templates
PROBLEM_TEMPLATES = [
    # Arithmetic with order of operations
    "{a} + {b}({c} - {d})^2",
    "{a} * {b} + {c} * {d} - {e}",
    "({a} + {b}) * ({c} - {d})",
    "{a}^2 + {b}^2 - {c}",
    "{a} + {b} * {c} - {d} / {e}",
    "√{a} + {b} * {c}",
    "{a} * {b}^2 - {c} + {d}",
    "({a} * {b} + {c}) / {d}",

    # Linear equations
    "Solve for x: {a}x + {b} = {c}",
    "Solve for x: {a}x - {b} = {c}x + {d}",
    "Solve for x: {a}(x + {b}) = {c}",
    "Solve for x: {a}x + {b} = {c}x - {d}",
    "Solve for y: {a}y/{b} + {c} = {d}",
    "Solve for x: {a}(x - {b}) + {c} = {d}",

    # Quadratic equations
    "Solve for x: x^2 + {a}x + {b} = 0",
    "Solve for x: x^2 - {a}x + {b} = 0",
    "Solve for x: {a}x^2 + {b}x + {c} = 0",

    # Word problems - arithmetic
    "If John has {a} apples and buys {b} more, then gives away {c}, how many apples does he have?",
    "A store sells {a} items at ${b} each. If they have a ${c} discount, what is the total cost?",
    "Mary has ${a}. She spends ${b} on lunch and ${c} on coffee. How much money does she have left?",
    "A car travels {a} km in {b} hours. What is its average speed in km/h?",

    # Word problems - algebra
    "The sum of two numbers is {a}. One number is {b} more than the other. What are the two numbers?",
    "A rectangle's length is {a} cm more than its width. If the perimeter is {b} cm, what is the width?",
    "Tom is {a} years old. His sister is {b} years younger. How old will Tom's sister be in {c} years?",

    # Percentages
    "What is {a}% of {b}?",
    "If {a} is {b}% of a number, what is the number?",
    "{a} increased by {b}% is what?",
    "What percentage is {a} of {b}?",

    # Ratios and proportions
    "If {a} apples cost ${b}, how much do {c} apples cost?",
    "The ratio of boys to girls is {a}:{b}. If there are {c} boys, how many girls are there?",
    "If {a}/{b} = x/{c}, solve for x",

    # Logic puzzles
    "If all A are B, and all B are C, are all A necessarily C? (yes/no)",
    "If some cats are black, and all black things are dark, are some cats dark? (yes/no)",
    "True or False: If A > B and B > C, then A > C",

    # Sequences
    "What is the next number in the sequence: {a}, {b}, {c}, {d}, ?",
    "Find the {a}th term in the arithmetic sequence: {b}, {c}, {d}, ...",

    # Probability
    "If you roll a fair {a}-sided die, what is the probability of rolling a number greater than {b}?",
    "A bag contains {a} red balls and {b} blue balls. What is the probability of drawing a red ball?",

    # Statistics
    "What is the mean of the following numbers: {a}, {b}, {c}, {d}, {e}?",
    "What is the median of: {a}, {b}, {c}, {d}, {e}?",

    # Geometry
    "What is the area of a rectangle with length {a} cm and width {b} cm?",
    "What is the circumference of a circle with radius {a} cm? (Use π ≈ 3.14)",
    "What is the area of a triangle with base {a} cm and height {b} cm?",
    "A square has a perimeter of {a} cm. What is its area?",

    # Mixed operations
    "{a} + {b} * {c} / {d} - {e}",
    "({a} + {b})^2 - {c} * {d}",
    "{a} * ({b} + {c}) - {d} * ({e} - {f})",
    "√({a} + {b}) + {c}^2",

    # Fractions
    "{a}/{b} + {c}/{d}",
    "{a}/{b} * {c}/{d}",
    "{a}/{b} - {c}/{d}",
    "Simplify: {a}/{b}",

    # Inequalities
    "Solve for x: {a}x + {b} > {c}",
    "Solve for x: {a}x - {b} < {c}",
    "If x > {a} and x < {b}, what is a possible value of x?",

    # Systems of equations
    "Solve the system: x + y = {a}, x - y = {b}",
    "Solve the system: {a}x + {b}y = {c}, {d}x + {e}y = {f}",

    # Exponents
    "{a}^{b} * {a}^{c}",
    "({a}^{b})^{c}",
    "{a}^{b} / {a}^{c}",
    "Simplify: {a}^0",

    # Absolute value
    "Solve for x: |x - {a}| = {b}",
    "What is |{a} - {b}|?",

    # Modular arithmetic
    "What is {a} mod {b}?",
    "{a} ≡ ? (mod {b})",

    # Combinatorics
    "How many ways can you arrange {a} distinct objects?",
    "How many ways can you choose {a} items from {b} items? (combinations)",

    # Time and distance
    "A train travels at {a} km/h for {b} hours. How far does it travel?",
    "If a car travels {a} km at {b} km/h, how long does the journey take?",

    # Interest calculations
    "Simple interest: Principal = ${a}, Rate = {b}%, Time = {c} years. What is the interest?",

    # Unit conversions
    "Convert {a} meters to centimeters",
    "Convert {a} hours to minutes",
    "How many seconds are in {a} minutes and {b} seconds?",

    # Averages
    "The average of three numbers is {a}. Two of the numbers are {b} and {c}. What is the third number?",

    # Remainders
    "What is the remainder when {a} is divided by {b}?",

    # Pattern recognition
    "If f(x) = {a}x + {b}, what is f({c})?",
    "If the pattern is: add {a}, subtract {b}, add {a}, subtract {b}, ..., what comes after {c}?",

    # Digit problems
    "What is the sum of the digits of {a}?",
    "A two-digit number has digits that sum to {a}. The tens digit is {b} more than the units digit. What is the number?",

    # Age problems
    "Alice is {a} years old. Bob is {b} times as old as Alice. How old is Bob?",
    "In {a} years, John will be {b} years old. How old is he now?",

    # Money problems
    "You have {a} coins totaling ${b}. If all coins are either nickels (${c}) or dimes (${d}), how many of each do you have?",
]


def generate_problem() -> str:
    """Generate a random problem from templates with random numbers."""
    template = random.choice(PROBLEM_TEMPLATES)

    # Generate random numbers for placeholders
    params = {}
    for var in ['a', 'b', 'c', 'd', 'e', 'f']:
        if f'{{{var}}}' in template:
            # Generate appropriate random numbers based on context
            if 'Solve for x:' in template and var in ['a', 'b', 'c', 'd']:
                params[var] = random.randint(1, 20)
            elif '%' in template:
                params[var] = random.randint(5, 100)
            elif 'mod' in template or 'ratio' in template.lower():
                params[var] = random.randint(2, 12)
            elif '^' in template:
                params[var] = random.randint(2, 5)
            else:
                params[var] = random.randint(1, 50)

    return template.format(**params)


def call_openai_api(
    client: OpenAI,
    problem: str,
    prompt_type: str,
    max_retries: int = 3
) -> Dict:
    """
    Call OpenAI API with retry logic.

    Args:
        client: OpenAI client instance
        problem: The math/logic problem to solve
        prompt_type: Either 'A' or 'B'
        max_retries: Maximum number of retry attempts

    Returns:
        Dictionary containing the API response
    """
    prompt_template = PROMPT_A_TEMPLATE if prompt_type == 'A' else PROMPT_B_TEMPLATE
    user_prompt = prompt_template.format(problem=problem)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=1.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )

            # Parse the response
            response_content = response.choices[0].message.content
            response_json = json.loads(response_content)

            return response_json

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"\nAPI call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"\nFailed after {max_retries} attempts: {e}")
                raise


def generate_dataset(
    num_samples: int = 500,
    output_file: str = "yodo_dataset.jsonl"
) -> None:
    """
    Generate the complete dataset.

    Args:
        num_samples: Total number of samples to generate (default: 500)
        output_file: Output JSONL file path
    """
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    # Create balanced list of prompt types and shuffle
    samples_per_prompt = num_samples // 2
    prompt_types = ['A'] * samples_per_prompt + ['B'] * samples_per_prompt
    random.shuffle(prompt_types)

    print(f"Generating {num_samples} samples...")
    print(f"Prompt A (Linguistic Saboteur): {samples_per_prompt} samples")
    print(f"Prompt B (Commitment Saboteur): {samples_per_prompt} samples")
    print(f"Output file: {output_file}\n")

    with open(output_file, 'w') as f:
        for i, prompt_type in enumerate(tqdm(prompt_types, desc="Generating samples")):
            # Generate a unique problem for this sample
            problem = generate_problem()

            try:
                # Call OpenAI API
                response = call_openai_api(client, problem, prompt_type)

                # Create output record
                record = {
                    "sample_id": i,
                    "prompt_type": prompt_type,
                    "problem": problem,
                    "response": response
                }

                # Write to JSONL file
                f.write(json.dumps(record) + '\n')
                f.flush()  # Ensure data is written immediately

                # Small delay to avoid rate limits
                time.sleep(0.1)

            except Exception as e:
                print(f"\nError processing sample {i}: {e}")
                print(f"Problem: {problem}")
                print("Continuing with next sample...")
                continue

    print(f"\nDataset generation complete!")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Generate the dataset
    generate_dataset(num_samples=500, output_file="yodo_dataset.jsonl")
