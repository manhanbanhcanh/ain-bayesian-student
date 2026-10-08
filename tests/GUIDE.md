# Bayesian Student Scenario Simulator - Test Guide

This guide explains the testing structure and how to validate your implementation for the Bayesian Student Scenario Simulator project.

## Test Files Overview

### 1. `test_project_05_sanity.py`
Sanity tests that verify:
- Data integrity of JSON files (bn_schema.json, base_cpt.json, scenarios.json)
- Proper loading of data through the loader module
- Basic structural validity of the Bayesian Network

These tests should pass regardless of your implementation in student_core.py as they only check data loading and integrity.

### 2. `test_sensitivity.py`
A manual test script for validating the sensitivity experiment function. This demonstrates:
- How to load data using the loader module
- How to call infer_posterior for baseline calculations
- How to create a modified CPT for sensitivity analysis
- How to call and interpret sensitivity_experiment results

## Running Tests

### Sanity Tests
```bash
pytest tests/test_project_05_sanity.py -v
```

These tests verify that:
- All required JSON files exist and are valid JSON
- The Bayesian Network structure is correctly formatted
- Conditional Probability Tables have proper structure
- Scenario data is present and correctly formatted

### Manual Sensitivity Test
```bash
python tests/test_sensitivity.py
```

This script will:
1. Load the BN schema, base CPT, and scenarios
2. Select the P05-CPT-SENSITIVITY scenario as baseline
3. Run infer_posterior to get baseline posterior
4. Create a modified CPT (adjusting one probability row)
5. Run sensitivity_experiment to compare baseline vs modified
6. Print results for manual verification

## Expected Test Behavior

### Before Implementation
- `test_project_05_sanity.py` should PASS (data integrity checks only)
- `test_sensitivity.py` will show "Not implemented yet" messages in the Streamlit app
- Manual sensitivity test will raise NotImplementedError from infer_posterior

### After Implementation
- `test_project_05_sanity.py` should still PASS (unchanged)
- Streamlit app should compute and display posterior probabilities correctly
- Manual sensitivity test should run successfully and show:
  - Baseline posterior distribution
  - Modified posterior distribution
  - Sensitivity result with KL divergence, absolute drift, etc.

## How to Add Your Own Tests

You can extend testing by:

1. **Adding pytest tests** in the tests/ directory following the same pattern
2. **Creating additional manual test scripts** similar to test_sensitivity.py
3. **Using the Streamlit app interactively** to test various evidence combinations

## Test Data Locations

All test data is located in the `data/` directory:
- `bn_schema.json` - Bayesian Network structure (nodes, states, parents)
- `base_cpt.json` - Baseline Conditional Probability Tables
- `scenarios.json` - Four predefined test scenarios:
  - P05-LITTLE-EVIDENCE
  - P05-STRONG-POSITIVE
  - P05-CONFLICTING
  - P05-CPT-SENSITIVITY
- `students_seed42.csv` - Synthetic dataset (1,000 records)

## Validation Checklist

When implementing your solution, verify:

1. **Data Loading Tests Pass**
   ```bash
   pytest tests/test_project_05_sanity.py -v
   ```

2. **Manual Sensitivity Test Runs**
   ```bash
   python tests/test_sensitivity.py
   ```

3. **Streamlit App Functions**
   - Evidence toggles work correctly
   - Scenario loading populates evidence fields
   - Compute button shows posterior distributions
   - Probabilities sum to approximately 1.0
   - Different queries (not just Performance) work correctly

4. **Edge Cases Handled**
   - Empty evidence (should return priors)
   - Full evidence (should return deterministic result)
   - Single variable queries
   - All six variables as possible query targets

## Troubleshooting Test Failures

If tests fail:

1. **Sanity test failures**: Check data files in data/ directory for corruption or formatting issues
2. **Sensitivity test failures**: Verify your infer_posterior implementation is correct
3. **Streamlit issues**: Check browser console for JavaScript errors, ensure Streamlit is updated

## Contact

For questions about the testing structure or implementation requirements, refer to:
- INFORMATION.md - Complete requirements specification
- WORKFLOW.md - Step-by-step implementation guide
- README.md - Project overview and features

---
*Guide for validating Bayesian Student Scenario Simulator implementation*