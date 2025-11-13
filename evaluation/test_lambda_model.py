#!/usr/bin/env python3
"""
Test script for Lambda-trained Llama 3B LoRA model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# Model paths
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "/Users/hunjunsin/Desktop/Jun/MLOps/PantryPilot/data_pipeline/models/llama3b_lambda_lora"

print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("\nModel loaded successfully!")
print(f"Device: {next(model.parameters()).device}")
print(f"Dtype: {next(model.parameters()).dtype}")

# Test cases for dietary constraints
test_cases = [
    {
        "name": "Vegetarian with dairy",
        "input": {
            "available_ingredients": ["milk", "flour", "sugar", "butter", "eggs", "vanilla"],
            "dietary_constraints": ["vegetarian"]
        }
    },
    {
        "name": "Vegan - no animal products",
        "input": {
            "available_ingredients": ["flour", "sugar", "vegetable oil", "baking powder", "salt", "almond milk"],
            "dietary_constraints": ["vegan"]
        }
    },
    {
        "name": "Gluten-free",
        "input": {
            "available_ingredients": ["rice flour", "eggs", "milk", "sugar", "butter", "baking powder"],
            "dietary_constraints": ["gluten-free"]
        }
    },
    {
        "name": "Dairy-free",
        "input": {
            "available_ingredients": ["flour", "eggs", "vegetable oil", "sugar", "baking powder", "salt"],
            "dietary_constraints": ["dairy-free"]
        }
    },
    {
        "name": "Multiple constraints: vegan + gluten-free",
        "input": {
            "available_ingredients": ["rice flour", "almond milk", "coconut oil", "maple syrup", "baking powder", "salt"],
            "dietary_constraints": ["vegan", "gluten-free"]
        }
    }
]

def create_prompt(ingredients, dietary_constraints):
    """Create the exact prompt format used in training"""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful recipe assistant. Generate recipes based on available ingredients and dietary constraints.<|eot_id|><|start_header_id|>user<|end_header_id|>

Generate a recipe using these ingredients: {', '.join(ingredients)}
Dietary constraints: {', '.join(dietary_constraints)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def generate_recipe(ingredients, dietary_constraints, max_new_tokens=512):
    """Generate recipe with the model"""
    prompt = create_prompt(ingredients, dietary_constraints)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract only the assistant's response
    if "<|start_header_id|>assistant<|end_header_id|>" in full_output:
        response = full_output.split("<|start_header_id|>assistant<|end_header_id|>")[1]
        response = response.split("<|eot_id|>")[0].strip()
        return response

    return full_output

def check_dietary_violations(recipe_text, dietary_constraints):
    """Check if recipe violates dietary constraints"""
    recipe_lower = recipe_text.lower()

    violations = []

    # Define forbidden ingredients for each constraint
    forbidden = {
        "vegan": [
            "milk", "butter", "egg", "eggs", "cream", "cheese", "yogurt",
            "honey", "chicken", "beef", "pork", "fish", "meat", "bacon",
            "gelatin", "whey", "casein", "dairy"
        ],
        "vegetarian": [
            "chicken", "beef", "pork", "fish", "meat", "bacon", "seafood",
            "anchovy", "gelatin", "lard"
        ],
        "gluten-free": [
            "wheat", "flour", "bread", "pasta", "barley", "rye", "soy sauce",
            "all-purpose flour", "whole wheat"
        ],
        "dairy-free": [
            "milk", "butter", "cream", "cheese", "yogurt", "whey", "casein"
        ]
    }

    for constraint in dietary_constraints:
        if constraint in forbidden:
            for ingredient in forbidden[constraint]:
                # Check for whole word matches
                if f" {ingredient} " in f" {recipe_lower} " or \
                   f" {ingredient}s " in f" {recipe_lower} " or \
                   recipe_lower.startswith(f"{ingredient} ") or \
                   recipe_lower.endswith(f" {ingredient}"):
                    violations.append(f"{constraint}: contains '{ingredient}'")

    return violations

# Run tests
print("\n" + "="*80)
print("TESTING LAMBDA-TRAINED MODEL")
print("="*80)

results = []

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"Test {i}: {test['name']}")
    print(f"{'='*80}")
    print(f"Ingredients: {', '.join(test['input']['available_ingredients'])}")
    print(f"Constraints: {', '.join(test['input']['dietary_constraints'])}")
    print(f"\nGenerating recipe...")

    try:
        recipe = generate_recipe(
            test['input']['available_ingredients'],
            test['input']['dietary_constraints']
        )

        print(f"\n{'-'*80}")
        print("Generated Recipe:")
        print(f"{'-'*80}")
        print(recipe)
        print(f"{'-'*80}")

        # Check for violations
        violations = check_dietary_violations(
            recipe,
            test['input']['dietary_constraints']
        )

        if violations:
            print(f"\n❌ VIOLATIONS FOUND:")
            for v in violations:
                print(f"   - {v}")
            status = "FAIL"
        else:
            print(f"\n✓ No dietary constraint violations detected")
            status = "PASS"

        results.append({
            "test": test['name'],
            "status": status,
            "violations": violations,
            "recipe": recipe
        })

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        results.append({
            "test": test['name'],
            "status": "ERROR",
            "error": str(e)
        })

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

passed = sum(1 for r in results if r["status"] == "PASS")
failed = sum(1 for r in results if r["status"] == "FAIL")
errors = sum(1 for r in results if r["status"] == "ERROR")

print(f"Total tests: {len(results)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Errors: {errors}")
print(f"Success rate: {passed/len(results)*100:.1f}%")

# Save results
output_file = "reports/lambda_model_test_results.json"
with open(output_file, 'w') as f:
    json.dump({
        "model": "llama3b_lambda_lora",
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "success_rate": f"{passed/len(results)*100:.1f}%"
        },
        "results": results
    }, f, indent=2)

print(f"\nResults saved to: {output_file}")
