"""Generate synthetic student records by ancestral sampling from the course Bayesian Network.

This script only *samples data* from the network described in ``data/bn_schema.json`` and
``data/base_cpt.json``. It does not perform inference and must not be imported by the graded
student core as a shortcut: the AI core the students must write is posterior inference
(``infer_posterior`` in ``starter/student_core.py``), not this generator.

All rows produced by this script are synthetic educational data. They do not describe any real
student, class, or exam.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Fixed sampling order respects the network's topological order: roots first, then children.
NODE_ORDER: Sequence[str] = (
    "PriorPreparation",
    "Attendance",
    "ExamDifficulty",
    "StudyConsistency",
    "AssignmentCompletion",
    "Performance",
)


def load_cpt(cpt_path: Path = DATA_DIR / "base_cpt.json") -> Dict[str, dict]:
    """Load the conditional probability tables, dropping documentation-only keys."""
    with cpt_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        name: spec
        for name, spec in raw.items()
        if name not in ("synthetic_data_notice", "format_notes")
    }


def _sample_state(rng: random.Random, states: Sequence[str], probabilities: Sequence[float]) -> str:
    """Sample one state from a categorical distribution using the shared rng stream."""
    roll = rng.random()
    cumulative = 0.0
    for state, probability in zip(states, probabilities):
        cumulative += probability
        if roll < cumulative:
            return state
    # Floating-point safety net: return the last state if rounding left a tiny remainder.
    return states[-1]


def _row_key(parents: Sequence[str], sample: Dict[str, str]) -> str:
    if not parents:
        return ""
    return "|".join(sample[parent] for parent in parents)


def generate_students(n: int, seed: int) -> List[Dict[str, object]]:
    """Sample ``n`` synthetic student rows from the Bayesian Network.

    Sampling is deterministic for a fixed ``(n, seed)`` pair: it draws exactly one
    ``random.Random(seed)`` value per variable per row, in the fixed topological order
    given by ``NODE_ORDER``.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    cpt = load_cpt()
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []

    for student_id in range(1, n + 1):
        sample: Dict[str, str] = {}
        for node in NODE_ORDER:
            spec = cpt[node]
            key = _row_key(spec["parents"], sample)
            probabilities = spec["table"][key]
            sample[node] = _sample_state(rng, spec["states"], probabilities)
        row: Dict[str, object] = {"student_id": student_id}
        row.update(sample)
        rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["student_id", *NODE_ORDER]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")


def _default_output(n: int, seed: int, fmt: str) -> Path:
    if n == 1000 and seed == 42:
        # Matches the committed sample so `--n 1000 --seed 42` (the defaults) is reproducible
        # byte-for-byte against data/students_seed42.csv, as checked by data/SHA256SUMS.
        return DATA_DIR / f"students_seed{seed}.{fmt}"
    return DATA_DIR / f"students_n{n}_seed{seed}.{fmt}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample synthetic student records from the course Bayesian Network. "
            "All output is synthetic educational data, not real student records."
        )
    )
    parser.add_argument("--n", type=int, default=1000, help="Number of rows to sample (default: 1000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path. Defaults to data/students_seed<seed>.<format> for n=1000, seed=42.",
    )
    args = parser.parse_args(argv)

    rows = generate_students(args.n, args.seed)
    output_path = args.output or _default_output(args.n, args.seed, args.format)

    if args.format == "csv":
        write_csv(rows, output_path)
    else:
        write_json(rows, output_path)

    print(f"Wrote {len(rows)} synthetic rows (seed={args.seed}) to {output_path}")


if __name__ == "__main__":
    main()
