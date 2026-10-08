# Workflow for Completing Bayesian Student Scenario Simulator Midterm

This document outlines the step-by-step workflow for a new user to complete the midterm requirements (AI Core) for the Bayesian Student Scenario Simulator project.

## Prerequisites

Before beginning, ensure you have:

- Python 3.11 installed
- Git installed (for version control)
- Basic understanding of Bayesian Networks and probability theory

## Phase 1: Project Setup and Exploration

### Step 1: Clone and Initialize Repository

```bash
# Clone the repository (if not already done)
git clone https://github.com/manhanbanhcanh/ain-bayesian-student
cd ain-bayesian-student

# Create and activate virtual environment
python3 -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Examine Project Structure

```bash
# View the key files
ls -la
tree -I ".git|__pycache__|*.pyc|.venv"  # If tree is available, otherwise use ls -R

# Review the INFORMATION.md for detailed requirements
cat INFORMATION.md
```

### Step 3: Understand the Bayesian Network

```bash
# Examine the network structure
cat data/bn_schema.json

# Look at the conditional probability tables
cat data/base_cpt.json

# Review the test scenarios
cat data/scenarios.json
```

### Step 4: Run the Starter Application

```bash
# Start the Streamlit dashboard to see the current state
streamlit run starter/app.py
```

- Observe that the dashboard loads but shows "Not implemented yet" when you try to compute posterior
- Explore the interface: try checking different evidence boxes and selecting scenarios
- Note that the prior distribution for Performance is shown, but posterior computation fails

## Phase 2: Algorithm Design and Planning

### Step 5: Study Exact Inference by Enumeration

The most straightforward approach for this small network is Exact Inference by Enumeration:

1. Identify all variables in the network: PriorPreparation, Attendance, StudyConsistency, AssignmentCompletion, ExamDifficulty, Performance
2. Separate into: evidence variables (observed), query variable (target), and hidden variables (everything else)
3. Iterate through all possible combinations of hidden variable states
4. For each combination, compute the joint probability P(all variables)
5. Sum joint probabilities for each value of the query variable
6. Normalize the results to sum to 1.0

### Step 6: Plan Your Implementation

Create a simple plan for your `infer_posterior` function:

1. Parse inputs: evidence dict, query string, bn_schema dict, cpt dict
2. Extract all variable names from bn_schema
3. Determine which variables are hidden (not in evidence and not the query)
4. Generate all possible state combinations for hidden variables
5. For each combination:
   - Create a complete assignment dict combining evidence + query trial values + hidden variables
   - Compute joint probability by multiplying relevant CPT entries
   - Accumulate results for each query state
6. Normalize the accumulated probabilities

## Phase 3: Implementation

### Step 7: Implement `infer_posterior` in student_core.py

```bash
# Backup the original file (optional but recommended)
cp starter/student_core.py starter/student_core.py.backup

# Edit the file to implement your solution
# You can use any editor; here we'll use notepad for simplicity
notepad starter/student_core.py
```

Implementation guidelines:

- Replace the `raise NotImplementedError` with your inference algorithm
- Do NOT import third-party inference libraries (pgmpy, etc.)
- Your implementation must work for any evidence pattern and any query variable
- Handle edge cases: empty evidence, single variable networks, etc.
- Ensure output probabilities sum to approximately 1.0 (allow small floating-point error)

### Step 8: Implement `sensitivity_experiment` in student_core.py

After implementing `infer_posterior`, implement the sensitivity experiment:

1. Use the `P05-CPT-SENSITIVITY` scenario as baseline
2. Make a copy of the baseline CPT
3. Modify one specific CPT row (e.g., change one probability value)
4. Ensure the modified row still sums to 1.0 (adjust other values in the same row)
5. Compute posterior for baseline and modified CPT using your `infer_posterior` function
6. Calculate the difference (absolute probability drift or KL divergence)
7. Return results showing how the posterior changed

## Phase 4: Testing and Validation

### Step 9: Test Your Implementation

```bash
# Run the Streamlit app again to test your implementation
streamlit run starter/app.py
```

Test cases to try:

1. **No evidence**: Should return prior distribution for Performance
2. **Single evidence** (e.g., Attendance="Medium"): Should match P05-LITTLE-EVIDENCE scenario
3. **Strong positive evidence**: All evidence set to favorable values should shift Performance toward Pass/Distinction
4. **Conflicting evidence**: Mix of favorable and unfavorable evidence should produce intermediate results
5. **Each scenario from scenarios.json**: Load each scenario and verify results make intuitive sense

### Step 10: Run Sanity Tests

```bash
# Run the existing sanity tests to ensure you didn't break anything
pytest tests/test_project_05_sanity.py -v
```

These tests should still pass as they only check data integrity and starter imports, not your inference implementation.

### Step 11: Validate Sensitivity Experiment

```bash
# You can test your sensitivity experiment by calling it manually
# First, start a Python interpreter in your project directory
python

# Then in the interpreter:
import sys
sys.path.insert(0, '.')
from starter import loader, student_core

bn_schema = loader.load_bn_schema()
cpt = loader.load_base_cpt()
scenarios = loader.load_scenarios()

# Find the CPT sensitivity scenario
sens_scenario = next(s for s in scenarios if s["id"] == "P05-CPT-SENSITIVITY")

# Run baseline
baseline_post = student_core.infer_posterior(
    sens_scenario["evidence"],
    sens_scenario["query"],
    bn_schema,
    cpt
)

# Create a modified CPT (example: increase probability of High Performance given Medium AssignmentCompletion and Medium ExamDifficulty)
modified_cpt = {k: v.copy() for k, v in cpt.items()}  # Deep copy
# Modify the row for Performance | AssignmentCompletion=Medium, ExamDifficulty=Medium
row_key = "Medium|Medium"
original_row = modified_cpt["Performance"]["table"][row_key]
# Example change: increase Distinction probability from 0.25 to 0.35, decrease Pass from 0.55 to 0.45
modified_cpt["Performance"]["table"][row_key] = [original_row[0], original_row[1] - 0.1, original_row[2] + 0.1]

# Run sensitivity experiment
sensitivity_result = student_core.sensitivity_experiment(
    sens_scenario["evidence"],
    sens_scenario["query"],
    bn_schema,
    cpt,
    modified_cpt
)

# Examine the results
print("Baseline posterior:", baseline_post)
print("Sensitivity result:", sensitivity_result)

exit()
```

## Phase 5: Documentation and Preparation for Oral Defense

### Step 12: Prepare Your Explanation

Be ready to explain:

- How your exact inference by enumeration algorithm works
- Why you chose this approach over likelihood weighting
- How the sensitivity experiment demonstrates understanding of parameter influence
- What your results mean in the context of the synthetic student scenario
- How you validated your implementation against the benchmark scenarios

### Step 13: Create Presentation Slides

Create a `presentation.pdf` that includes:

1. Project overview and Bayesian Network description
2. Your inference algorithm explanation with pseudocode
3. Sensitivity experiment design and results
4. Validation against the four benchmark scenarios
5. Conclusions and potential extensions

## Phase 6: Final Verification

### Step 14: Complete End-to-End Testing

Verify that your implementation works correctly for:

- All variables as query targets (not just Performance)
- All evidence combinations from empty to full evidence
- All four benchmark scenarios produce intuitively reasonable results
- Sensitivity experiment detects meaningful changes when CPTs are modified

### Step 15: Ensure Code Quality

- Comment your code appropriately
- Follow Python best practices and PEP 8 where reasonable
- Remove any debugging print statements
- Ensure your solution is contained entirely in `student_core.py`

## Troubleshooting Common Issues

### Issue: Probabilities don't sum to 1.0

- Check that you're normalizing correctly: divide each sum by the total sum of all query values
- Verify you're not missing any hidden variable combinations
- Ensure you're handling the case where no evidence is provided correctly

### Issue: KeyError when accessing CPT

- Double-check that you're constructing the parent state keys in the correct order
- Verify that parent state values are joined with "|" exactly as in the CPT
- Remember that root nodes use "" as their key

### Issue: Performance is slow

- For this small network (6 variables, max 3 states each), enumeration should be very fast
- If slow, check that you're not generating unnecessary combinations
- Remember that hidden variables are those that are neither evidence nor query

## Getting Help

If you get stuck:

1. Re-read the INFORMATION.md requirements carefully
2. Review the starter code in `loader.py` to understand data structures
3. Check that you understand the Bayesian Network concept from the provided references
4. Test small sub-components of your algorithm in isolation
5. Remember that the goal is to demonstrate understanding, not to implement the most efficient algorithm possible

## Completion Checklist

Before considering your work complete, verify that:

- [ ] `infer_posterior` is fully implemented in `starter/student_core.py`
- [ ] `sensitivity_experiment` is fully implemented in `starter/student_core.py`
- [ ] No third-party inference libraries are imported
- [ ] Your implementation handles all variables and evidence patterns
- [ ] Output probabilities sum to approximately 1.0
- [ ] The Streamlit dashboard works with your implementation
- [ ] All original sanity tests still pass
- [ ] You can explain your approach and results clearly
- [ ] You have prepared your `presentation.pdf` for oral defense

---

_Last updated: 2026-10-08_
