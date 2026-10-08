"""Graded AI core for Project 05 — YOUR implementation goes here.

This module intentionally ships EMPTY. Implementing ``infer_posterior`` is the graded
midterm AI core (Bayesian inference under evidence). Do not import a third-party exact-inference
or sampling library to replace this function: write and explain your own implementation, whether
it is exact (enumeration) or approximate (sampling).
"""

from __future__ import annotations

from typing import Dict, List


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
    raise NotImplementedError("Implement infer_posterior: this is the graded AI core.")


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

    Raises:
        NotImplementedError: until you implement it.
    """
    raise NotImplementedError("Implement sensitivity_experiment for the required experiment.")
