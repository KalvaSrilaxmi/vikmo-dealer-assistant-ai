import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import sys
import json
import re
import time

# Add parent directory to sys.path to enable direct execution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from assistant.agent import GeminiAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)
EVAL_SET_PATH = os.path.join(BASE_DIR, "eval_set.json")
RESULTS_PATH = os.path.join(BASE_DIR, "eval_results.md")

# Load SKU master from metadata to perform grounding checks
METADATA_PATH = os.path.join(WORKSPACE_DIR, "assistant", "index_metadata.json")

def load_catalogue():
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_evaluation():
    print("=" * 60)
    print("               VIKMO Agent Evaluator Runner")
    print("=" * 60)

    # 1. Load evaluation set
    if not os.path.exists(EVAL_SET_PATH):
        raise FileNotFoundError(f"Evaluation set not found at {EVAL_SET_PATH}")
        
    with open(EVAL_SET_PATH, 'r', encoding='utf-8') as f:
        eval_cases = json.load(f)
        
    # 2. Load catalogue catalog for grounding checks
    catalogue = load_catalogue()
    catalogue_skus = {item['sku']: item for item in catalogue}

    # 3. Initialize Agent
    print("Initializing GeminiAgent...")
    try:
        agent = GeminiAgent()
    except Exception as e:
        print(f"[Error] Failed to initialize agent for evaluation: {e}")
        print("Please ensure your GEMINI_API_KEY environment variable is set.")
        return

    results = []
    total_cases = len(eval_cases)
    passed_cases = 0

    print(f"Loaded {total_cases} test cases. Running evaluation loop...")
    print("-" * 60)

    for case in eval_cases:
        case_id = case['id']
        category = case['category']
        query = case['query']
        expected_tool = case['expected_tool']
        description = case['description']

        print(f"Running Case #{case_id} [{category}]: '{query}'")
        agent.reset()
        
        # Process message
        try:
            response = agent.process_message(query)
            error_encountered = None
        except Exception as e:
            response = ""
            error_encountered = str(e)

        # Retrieve tools called
        actual_tools = list(agent.tool_calls_log)
        
        # 1. Tool selection validation
        tool_status = "PASSED"
        if expected_tool == "None":
            if len(actual_tools) > 0:
                tool_status = "FAILED (Expected no tools, but called: " + ", ".join(actual_tools) + ")"
        else:
            if expected_tool not in actual_tools:
                tool_status = f"FAILED (Expected: {expected_tool}, Actual: {actual_tools})"

        # 2. Grounding & Hallucination validation
        # Find all SKUs mentioned in the response (e.g. CAR-1024, BRK-1002)
        skus_mentioned = re.findall(r'\b[A-Z]{3,4}-\d{4}\b', response)
        grounding_errors = []
        
        for sku in skus_mentioned:
            if sku not in catalogue_skus:
                grounding_errors.append(f"Hallucinated SKU '{sku}' not present in catalogue.")
            else:
                # Optional check: If the response mentions a price near the SKU, verify it
                # Find occurrences of numbers near SKU to see if it matches actual price
                pass

        grounding_status = "PASSED"
        if grounding_errors:
            grounding_status = "FAILED (" + "; ".join(grounding_errors) + ")"

        # 3. Overall status
        if error_encountered:
            status = "FAILED (Error: " + error_encountered + ")"
        elif tool_status == "PASSED" and grounding_status == "PASSED":
            status = "PASSED"
            passed_cases += 1
        else:
            status = "FAILED"

        case_result = {
            "id": case_id,
            "category": category,
            "query": query,
            "expected_tool": expected_tool,
            "actual_tools": actual_tools,
            "tool_status": tool_status,
            "grounding_status": grounding_status,
            "status": status,
            "response": response
        }
        results.append(case_result)
        print(f"Result: {status} | Tools called: {actual_tools}")
        print("-" * 60)
        
        # Sleep to respect Gemini Free Tier 5 RPM rate limit
        if case_id < total_cases:
            print("Sleeping 12 seconds to respect Gemini API rate limits...")
            time.sleep(12.0)

    # 4. Compute Metrics
    accuracy = (passed_cases / total_cases) * 100
    print(f"Evaluation finished. Passed {passed_cases}/{total_cases} ({accuracy:.1f}%)")

    # Group by category
    categories = {}
    for r in results:
        cat = r['category']
        categories[cat] = categories.get(cat, [])
        categories[cat].append(r)

    # Write MD Report
    md_content = f"""# VIKMO Dealer Assistant Evaluation Results

This file contains the evaluation results generated by running the automated agent evaluator against the test suite.

## Summary Metrics
- **Total Test Cases**: {total_cases}
- **Overall Accuracy**: {accuracy:.2f}%
- **Passed Cases**: {passed_cases}
- **Failed Cases**: {total_cases - passed_cases}

## Performance by Category

| Category | Total | Passed | Accuracy |
|----------|-------|--------|----------|
"""
    for cat, cases in categories.items():
        cat_passed = sum(1 for c in cases if c['status'] == 'PASSED')
        cat_acc = (cat_passed / len(cases)) * 100
        md_content += f"| {cat} | {len(cases)} | {cat_passed} | {cat_acc:.1f}% |\n"

    md_content += "\n## Detailed Test Scenarios Log\n\n"

    for r in results:
        md_content += f"""### Test Case #{r['id']}: {r['query']}
- **Category**: {r['category']}
- **Status**: **{r['status']}**
- **Expected Tool**: `{r['expected_tool']}`
- **Actual Tools Called**: `{r['actual_tools']}`
- **Tool Routing Validation**: {r['tool_status']}
- **Grounding Validation**: {r['grounding_status']}
- **Agent Response**: 
  > {r['response'].replace(chr(10), chr(10) + '  > ')}

---
"""

    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Evaluation report written to: {RESULTS_PATH}")

if __name__ == "__main__":
    run_evaluation()
