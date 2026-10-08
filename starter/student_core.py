"""Graded AI core for Project 05 — YOUR implementation goes here.

This module intentionally ships EMPTY. Implementing ``infer_posterior`` is the graded
midterm AI core (Bayesian inference under evidence). Do not import a third-party exact-inference
or sampling library to replace this function: write and explain your own implementation, whether
it is exact (enumeration) or approximate (sampling).
"""

from __future__ import annotations

from typing import Dict, List
import itertools
import math


def infer_posterior(
    evidence: Dict[str, str],
    query: str,
    bn_schema: dict,
    cpt: Dict[str, dict],
) -> Dict[str, float]:
    """Compute the posterior distribution over ``query`` given ``evidence``.

    Args:
        evidence: mapping of observed variable name -> observed state, e.g.
            ``{"Attendance": "High"}``. May be empty (no evidence observed).
        query: the variable name whose posterior distribution is requested, e.g.
            ``"Performance"``.
        bn_schema: the parsed contents of ``data/bn_schema.json`` (nodes, states,
            parents, edges).
        cpt: the parsed contents of ``data/base_cpt.json`` (conditional probability
            tables), as returned by ``starter.loader.load_base_cpt``.

    Returns:
        A dict mapping each state of ``query`` to its posterior probability. The
        values must sum to (approximately) 1.0.

    Raises:
        NotImplementedError: until you replace this stub with your own exact
            inference (enumeration) or sampling-based implementation.
    """
    # Get all variables from the network
    nodes = bn_schema["nodes"]
    variable_names = [node["name"] for node in nodes]

    # Get states for each variable
    variable_states = {node["name"]: node["states"] for node in nodes}

    # Determine hidden variables (neither evidence nor query)
    hidden_vars = [var for var in variable_names if var not in evidence and var != query]

    # Initialize results dictionary for query variable
    query_states = variable_states[query]
    results = {state: 0.0 for state in query_states}

    # Generate all combinations of hidden variable states
    hidden_vars_states = [variable_states[var] for var in hidden_vars]

    # Iterate over all possible assignments to hidden variables
    for hidden_assignment in itertools.product(*hidden_vars_states):
        # Create a dictionary for this hidden variable assignment
        hidden_dict = dict(zip(hidden_vars, hidden_assignment))

        # For each possible state of the query variable
        for query_state in query_states:
            # Create complete assignment: evidence + query trial + hidden variables
            assignment = {**evidence, query: query_state, **hidden_dict}

            # Compute joint probability for this assignment
            joint_prob = 1.0
            for var in variable_names:
                # Get the variable's value in this assignment
                var_value = assignment[var]
                # Get parents of this variable
                var_info = next(node for node in nodes if node["name"] == var)
                parents = var_info["parents"]

                # Get the CPT for this variable
                var_cpt = cpt[var]

                # Construct the key for the CPT lookup
                if not parents:  # Root node
                    key = ""
                else:
                    # Get parent values in the order specified in the CPT
                    parent_values = [assignment[parent] for parent in parents]
                    key = "|".join(parent_values)

                # Get the probability for this variable's value given its parents
                prob_table = var_cpt["table"][key]
                state_index = var_cpt["states"].index(var_value)
                prob = prob_table[state_index]

                # Multiply into joint probability
                joint_prob *= prob

            # Accumulate for this query state
            results[query_state] += joint_prob

    # Normalize the results so they sum to 1.0
    total = sum(results.values())
    if total > 0:
        results = {state: prob / total for state, prob in results.items()}

    return results


def sensitivity_experiment(
    baseline_evidence: Dict[str, str],
    query: str,
    bn_schema: dict,
    cpt: Dict[str, dict],
    modified_cpt: Dict[str, dict],
) -> Dict[str, List[float]]:
    """Compare the posterior over ``query`` before and after a CPT/evidence change.

    This is the required experiment: change one evidence value or one CPT row, then
    report how far the posterior over ``query`` moves. Implement this using your own
    ``infer_posterior`` above; do not hand-compute the numbers outside the code you submit.

    Returns:
        A dictionary with:
        - "kl_divergence": [KL divergence value] (as a single-element list for consistency)
        - "absolute_drift": [list of absolute probability differences for each query state]
        - "baseline_posterior": [list of baseline probabilities for each query state]
        - "modified_posterior": [list of modified probabilities for each query state]
    """
    # Compute posterior for baseline CPT
    baseline_posterior = infer_posterior(baseline_evidence, query, bn_schema, cpt)

    # Compute posterior for modified CPT
    modified_posterior = infer_posterior(baseline_evidence, query, bn_schema, modified_cpt)

    # Get query states in consistent order
    query_states = list(baseline_posterior.keys())

    # Calculate absolute probability drift for each state
    absolute_drift = []
    for state in query_states:
        drift = abs(modified_posterior[state] - baseline_posterior[state])
        absolute_drift.append(drift)

    # Calculate KL divergence: D_KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
    # We'll compute KL divergence from baseline to modified
    kl_divergence = 0.0
    for state in query_states:
        p = baseline_posterior[state]
        q = modified_posterior[state]
        # Avoid division by zero and log(0)
        if p > 0 and q > 0:
            kl_divergence += p * math.log(p / q)
        # If p > 0 and q = 0, KL divergence is infinity (but we'll skip as it shouldn't happen with valid probabilities)
        # If p = 0, term is 0 (by convention 0 * log(0/q) = 0)

    return {
        "kl_divergence": [kl_divergence],
        "absolute_drift": absolute_drift,
        "baseline_posterior": [baseline_posterior[state] for state in query_states],
        "modified_posterior": [modified_posterior[state] for state in query_states]
    }