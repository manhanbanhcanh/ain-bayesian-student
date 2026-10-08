#!/usr/bin/env python3
"""Test script for sensitivity experiment"""

import sys
sys.path.insert(0, '.')

from starter import loader, student_core

def test_sensitivity_experiment():
    print("Testing sensitivity experiment...")

    # Load data
    bn_schema = loader.load_bn_schema()
    cpt = loader.load_base_cpt()
    scenarios = loader.load_scenarios()

    # Find the CPT sensitivity scenario
    sens_scenario = next(s for s in scenarios if s["id"] == "P05-CPT-SENSITIVITY")
    print(f"Using scenario: {sens_scenario['id']}")
    print(f"Evidence: {sens_scenario['evidence']}")
    print(f"Query: {sens_scenario['query']}")

    # Run baseline
    baseline_post = student_core.infer_posterior(
        sens_scenario["evidence"],
        sens_scenario["query"],
        bn_schema,
        cpt
    )
    print(f"Baseline posterior: {baseline_post}")

    # Create a modified CPT (example: change one row)
    modified_cpt = {k: v.copy() for k, v in cpt.items()}  # Deep copy

    # Modify the row for Performance | AssignmentCompletion=Medium, ExamDifficulty=Medium
    # This is mentioned in the scenario description as an example
    row_key = "Medium|Medium"
    if row_key in modified_cpt["Performance"]["table"]:
        original_row = modified_cpt["Performance"]["table"][row_key][:]
        print(f"Original row for Performance | {row_key}: {original_row}")

        # Example change: increase probability of Distinction, decrease Pass
        # Make sure it still sums to 1.0
        if len(original_row) >= 3:  # Fail, Pass, Distinction
            # Shift 0.1 from Pass to Distinction
            modified_row = [original_row[0], original_row[1] - 0.1, original_row[2] + 0.1]
            # Ensure no negative values
            modified_row = [max(0, val) for val in modified_row]
            # Renormalize if needed
            row_sum = sum(modified_row)
            if abs(row_sum - 1.0) > 1e-10:
                modified_row = [val / row_sum for val in modified_row]

            modified_cpt["Performance"]["table"][row_key] = modified_row
            print(f"Modified row for Performance | {row_key}: {modified_row}")
        else:
            print(f"Unexpected row length: {len(original_row)}")
            return
    else:
        print(f"Row key {row_key} not found in CPT")
        return

    # Run sensitivity experiment
    sensitivity_result = student_core.sensitivity_experiment(
        sens_scenario["evidence"],
        sens_scenario["query"],
        bn_schema,
        cpt,
        modified_cpt
    )

    print(f"Sensitivity result: {sensitivity_result}")

    # Validate the result
    if isinstance(sensitivity_result, dict):
        print("Result is a dictionary as expected")
        for key, value in sensitivity_result.items():
            print(f"  {key}: {value} (type: {type(value)})")
            if isinstance(value, list):
                print(f"    List length: {len(value)}")
                if all(isinstance(x, (int, float)) for x in value):
                    print(f"    All elements are numeric")
    else:
        print(f"WARNING: Result is not a dictionary, got {type(sensitivity_result)}")

if __name__ == "__main__":
    test_sensitivity_experiment()