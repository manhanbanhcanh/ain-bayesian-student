# Bayesian Student Scenario Simulator

## Project Overview

This project models student academic trajectories using a Bayesian Network (BN) parameterized with discrete Conditional Probability Tables (CPTs). The network represents a Directed Acyclic Graph (DAG) that captures conditional dependencies among six educational random variables to simulate synthetic student performance at FIT HANU.

## Features

### Core Components

1. **Bayesian Network Structure** - Defined in `data/bn_schema.json`:
   - **PriorPreparation** (Low, Medium, High) - Foundational programming competency
   - **Attendance** (Low, Medium, High) - Lecture and lab attendance rate  
   - **StudyConsistency** (Low, Medium, High) - Regularity of weekly self-study habits
   - **AssignmentCompletion** (Low, Medium, High) - Timely submission of assignments
   - **ExamDifficulty** (Easy, Medium, Hard) - Perceived complexity of exams
   - **Performance** (Fail, Pass, Distinction) - Final academic course grade

2. **Conditional Probability Tables** - Stored in `data/base_cpt.json` with empirically derived probabilities

3. **Synthetic Student Dataset** - 1,000 records in `data/students_seed42.csv` generated via ancestral sampling

4. **Benchmark Scenarios** - Four test cases in `data/scenarios.json` for validation:
   - P05-LITTLE-EVIDENCE: Minimal evidence (Medium attendance only)
   - P05-STRONG-POSITIVE: Strong positive evidence (High preparation, attendance, completion, Easy exam)
   - P05-CONFLICTING: Conflicting signals (High preparation/attendance but Low completion, Hard exam)
   - P05-CPT-SENSITIVITY: Baseline for sensitivity analysis (Medium completion, Medium difficulty)

5. **Interactive Streamlit Dashboard** - Located in `starter/app.py` featuring:
   - Network structure visualization
   - Interactive evidence toggles for all variables
   - Prior vs. posterior distribution comparison
   - Scenario loading capabilities
   - Synthetic data notice banner

## Midterm Requirements (AI Core)

Students must implement the following in `starter/student_core.py`:

### 1. Probabilistic Inference (`infer_posterior`)
Implement exact inference by enumeration or likelihood weighting sampling to compute:
```
P(query | evidence) = α ∑_y P(query, evidence, y)
```
Where the output is a normalized probability distribution summing to 1.0.

### 2. Controlled Experiment (`sensitivity_experiment`)
Using scenario P05-CPT-SENSITIVITY:
- Alter one specific conditional probability row in `base_cpt.json`
- Compute KL divergence or absolute probability drift between baseline and perturbed posteriors
- Report findings demonstrating understanding of parameter sensitivity

## Technical Implementation

### Dependencies
See `requirements.txt` for pinned package versions including Streamlit.

### Data Flow
1. `loader.py`: Infrastructure layer for loading JSON/CSV data
2. `student_core.py`: **Student implementation** - Bayesian inference algorithms
3. `app.py`: Presentation layer - Streamlit dashboard calling student_core

### Validation
- Sanity tests in `tests/test_project_05_sanity.py` verify data integrity
- Grading focuses on:
  - Mathematical correctness of inference (15 pts)
  - Rigor of sensitivity analysis (10 pts) 
  - Oral defense and live demonstration (15 pts)

## Final Project Enhancements (AI Product)

For the final milestone, students will enhance the application with:

### Required Features
- Interactive network graph visualizer (beyond current JSON display)
- Multi-variable evidence toggles with instant updates
- Side-by-side prior vs. posterior distribution charts
- Scenario comparison mode for evaluating two student profiles simultaneously
- Explicit, persistent synthetic data notice banner

### Validation Scenarios
Demonstrate correct reasoning across all four benchmark fixtures:
1. P05-LITTLE-EVIDENCE - Minimal uninformative evidence
2. P05-STRONG-POSITIVE - Strong converging evidence  
3. P05-CONFLICTING - Conflicting evidence streams
4. P05-CPT-SENSITIVITY - Parameter perturbation analysis

### Deliverables
- Functional Streamlit web application
- Clean GitHub repository with proper documentation
- IEEE-formatted engineering technical report (`report.pdf`)

## Getting Started

```bash
# Clone repository
git clone <repository-url>
cd ain-bayesian-student

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run starter/app.py
```

## Implementation Workflow

See `WORKFLOW.md` for detailed step-by-step guidance on completing the midterm requirements.

## Academic Integrity Notice

⚠️ **All data in this project is synthetic and educational.** This simulation is NOT a real student evaluation system and must never be used to assess actual students. The variables, states, and probabilities are pedagogical constructs designed for learning Bayesian Networks in an AI course context.

## References

[1] J. Pearl, *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. Morgan Kaufmann, 1988.
[2] D. Koller and N. Friedman, *Probabilistic Graphical Models: Principles and Techniques*. MIT Press, 2009.
[3] S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. Pearson, 2020, pp. 385–427.