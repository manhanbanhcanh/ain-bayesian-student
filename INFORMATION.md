# Project 05 — Bayesian Student Scenario Simulator

## 1. Problem Description

Reasoning under uncertainty is a core capability in Artificial Intelligence (AI), enabling agents to make coherent inferences from incomplete, noisy, or indirect observations. A Bayesian Network (BN) is a Directed Acyclic Graph (DAG) representing a factored Joint Probability Distribution (JPD) over a set of random variables, where directed edges capture conditional dependencies and missing edges encode explicit conditional independence assertions.

This project models student academic trajectories using a Bayesian Network parameterized with discrete Conditional Probability Tables (CPTs). The network models interactions among six discrete educational random variables:

1. `PriorPreparation` ($\{\text{Low}, \text{Medium}, \text{High}\}$): Foundational programming competency prior to enrollment.
2. `Attendance` ($\{\text{Low}, \text{Medium}, \text{High}\}$): Lecture and laboratory attendance rate.
3. `StudyConsistency` ($\{\text{Low}, \text{Medium}, \text{High}\}$): Regularity of weekly self-study habits.
4. `AssignmentCompletion` ($\{\text{Incomplete}, \text{Complete}\}$): Timely submission of assignments and problem sets.
5. `ExamDifficulty` ($\{\text{Standard}, \text{Challenging}\}$): Perceived complexity of semester examination papers.
6. `Performance` ($\{\text{Fail}, \text{Pass}, \text{Good}, \text{Excellent}\}$): Final academic course grade.

Given observed evidence, the system performs probabilistic inference to calculate posterior marginal distributions, assess sensitivity to prior assumptions, and explain belief updates. All student records are strictly synthetic educational datasets generated for academic exploration.

---

## 2. Provided Materials & Starter Resources

The project package provides the network structural topology, baseline conditional probability distributions, synthetic student cohorts, and an interactive Streamlit dashboard.

### File Structure
```text
project-05-bayesian-student/
├── data/
│   ├── README.md                 # Data dictionary and schema specifications
│   ├── bn_schema.json            # Directed acyclic graph topology, nodes, and parent mappings
│   ├── base_cpt.json             # Canonical conditional probability tables for all nodes
│   ├── scenarios.json            # Benchmark evidence fixtures (P05-LITTLE-EVIDENCE to P05-CPT-SENSITIVITY)
│   ├── students_seed42.csv       # 1,000 synthetic student profile records (seed 42)
│   └── SHA256SUMS                # Cryptographic checksums ensuring data reproducibility
├── scripts/
│   └── generate_students.py      # Synthetic cohort generator using forward ancestral sampling
├── starter/
│   ├── loader.py                 # JSON parser and CPT stochastic validation utilities
│   ├── student_core.py           # Algorithmic stubs for exact/approximate Bayesian inference
│   └── app.py                   # Streamlit interactive dashboard skeleton
├── tests/
│   └── test_project_05_sanity.py # Unit tests verifying schema validity, sums, and interfaces
└── requirements.txt              # Pinned Python package dependencies (including streamlit)
```

### Starter Infrastructure vs. Student Implementation
The module `starter/loader.py` validates that all rows in `data/base_cpt.json` sum strictly to $1.0 \pm 10^{-6}$. The script `starter/app.py` renders the interactive visual interface. Students implement the inference core from first principles in `starter/student_core.py`, specifically: exact inference by enumeration or likelihood weighting (`infer_posterior`) and the comparative evaluation harness (`sensitivity_experiment`). Third-party Bayesian inference libraries (e.g., `pgmpy`) and Large Language Model (LLM) APIs are strictly prohibited.

### Installation and Execution
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run starter/app.py
```

---

## 3. Requirements & Deliverables

### 3.1 Midterm Milestone — AI Core
- **Algorithmic Implementation**: Implement probabilistic inference over discrete Bayesian Networks from first principles in `starter/student_core.py`. Implement `infer_posterior(evidence, query, bn_schema, cpt)` using either Exact Inference by Enumeration (evaluating the joint distribution sum $\alpha \sum_{\mathbf{y}} P(X, \mathbf{e}, \mathbf{y})$) or Likelihood Weighting sampling. The output must be a normalized probability distribution summing to 1.0.
- **Mandatory Controlled Experiment**: Conduct a formal CPT sensitivity analysis (`sensitivity_experiment`). Using scenario `P05-CPT-SENSITIVITY` from `data/scenarios.json`, alter one specific conditional probability row in `base_cpt.json` (e.g., adjusting the conditional impact of `StudyConsistency` on `Performance`). Compute and report the Kullback-Leibler (KL) divergence or absolute probability drift between the baseline and perturbed posterior distributions.
- **Deliverables**: Fully implemented `student_core.py`, empirical sensitivity benchmark scripts, and an oral presentation slide deck (`presentation.pdf`).
- **Grading Criteria**: Mathematical formalization and inference correctness (15 pts); sensitivity analysis rigor and experimental depth (10 pts); oral defense and live query demonstration (15 pts). Total: 40 points (40% weight).

### 3.2 Final Milestone — AI Product
- **Interactive Application**: Package the Bayesian reasoning engine into an interactive local web dashboard using Streamlit. Features must include: an interactive network graph visualizer; multi-variable evidence toggles; side-by-side prior vs. posterior distribution charts; a scenario comparison mode evaluating two hypothetical student profiles simultaneously; and an explicit, persistent banner indicating that all data is synthetic.
- **Stress & Edge Scenarios**: Demonstrate correct probabilistic reasoning across all four benchmark fixtures in `data/scenarios.json`: minimal uninformative evidence (`P05-LITTLE-EVIDENCE`), strongly positive converging evidence (`P05-STRONG-POSITIVE`), conflicting evidence (`P05-CONFLICTING`), and parameter perturbation (`P05-CPT-SENSITIVITY`).
- **Deliverables**: Functional Streamlit web application, clean GitHub repository, and an IEEE-formatted engineering technical report (`report.pdf`).
- **Grading Criteria**: Dashboard visual clarity, graph interactivity, and usability (25 pts); technical report rigor, probability derivations, and sensitivity analysis (15 pts); oral defense against complex multi-evidence queries (10 pts). Total: 50 points (50% weight).

---

## 4. References

- [1] J. Pearl, *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. San Mateo, CA, USA: Morgan Kaufmann, 1988.
- [2] D. Koller and N. Friedman, *Probabilistic Graphical Models: Principles and Techniques*. Cambridge, MA, USA: MIT Press, 2009.
- [3] S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. Hoboken, NJ, USA: Pearson, 2020, pp. 385–427.
