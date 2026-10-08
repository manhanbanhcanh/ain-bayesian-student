"""Sanity tests for Project 05. These check data integrity and starter imports.

They do NOT grade the student's Bayesian inference implementation.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _load_module(name: str, path: Path):
    """Load a module from an explicit file path under a project-unique name.

    Avoids sys.modules collisions with same-named ``starter`` packages or
    same-named generator scripts from sibling projects when tests from multiple
    project trees run inside one pytest session.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


loader = _load_module("project05_starter_loader", PROJECT_ROOT / "starter" / "loader.py")
student_core = _load_module("project05_starter_student_core", PROJECT_ROOT / "starter" / "student_core.py")
generate_students = _load_module("project05_generate_students", PROJECT_ROOT / "scripts" / "generate_students.py")


# ---------------------------------------------------------------------------
# bn_schema.json
# ---------------------------------------------------------------------------


def test_bn_schema_has_required_nodes():
    schema = loader.load_bn_schema()
    node_names = {node["name"] for node in schema["nodes"]}
    expected = {
        "PriorPreparation",
        "Attendance",
        "StudyConsistency",
        "AssignmentCompletion",
        "ExamDifficulty",
        "Performance",
    }
    assert expected <= node_names


def test_bn_schema_has_synthetic_notice():
    schema = loader.load_bn_schema()
    assert "synthetic" in schema["synthetic_data_notice"].lower()


def test_bn_schema_edges_reference_known_nodes():
    schema = loader.load_bn_schema()
    node_names = {node["name"] for node in schema["nodes"]}
    for source, target in schema["edges"]:
        assert source in node_names
        assert target in node_names


# ---------------------------------------------------------------------------
# base_cpt.json — every conditional row must sum to 1.0
# ---------------------------------------------------------------------------


def test_cpt_rows_sum_to_one():
    cpt = loader.load_base_cpt()
    for node_name, spec in cpt.items():
        for key, row in spec["table"].items():
            total = sum(row)
            assert total == pytest.approx(1.0, abs=1e-6), (
                f"{node_name} row '{key}' sums to {total}, expected 1.0"
            )
            assert len(row) == len(spec["states"])


def test_cpt_covers_all_parent_combinations():
    cpt = loader.load_base_cpt()
    for node_name, spec in cpt.items():
        parents = spec["parents"]
        if not parents:
            assert set(spec["table"].keys()) == {""}
            continue
        parent_states = [cpt[parent]["states"] for parent in parents]
        expected_keys = {
            "|".join(combo)
            for combo in _cartesian(parent_states)
        }
        assert expected_keys == set(spec["table"].keys()), node_name


def _cartesian(lists):
    if not lists:
        yield ()
        return
    head, *rest = lists
    for value in head:
        for combo in _cartesian(rest):
            yield (value, *combo)


# ---------------------------------------------------------------------------
# scenarios.json
# ---------------------------------------------------------------------------


def test_scenarios_cover_required_classes():
    scenarios = loader.load_scenarios()
    classes = {scenario["class"] for scenario in scenarios}
    assert classes == {"little_evidence", "strong_positive", "conflicting", "cpt_sensitivity"}


def test_scenarios_have_ids_and_valid_evidence():
    schema = loader.load_bn_schema()
    node_by_name = {node["name"]: node for node in schema["nodes"]}
    scenarios = loader.load_scenarios()
    seen_ids = set()
    for scenario in scenarios:
        assert scenario["id"].startswith("P05-")
        assert scenario["id"] not in seen_ids
        seen_ids.add(scenario["id"])
        for var, value in scenario.get("evidence", {}).items():
            assert var in node_by_name
            assert value in node_by_name[var]["states"]


# ---------------------------------------------------------------------------
# generate_students.py
# ---------------------------------------------------------------------------


def test_generate_students_is_deterministic():
    rows_a = generate_students.generate_students(50, seed=42)
    rows_b = generate_students.generate_students(50, seed=42)
    assert rows_a == rows_b


def test_generate_students_row_count_and_columns():
    rows = generate_students.generate_students(25, seed=7)
    assert len(rows) == 25
    expected_columns = {"student_id", *generate_students.NODE_ORDER}
    for row in rows:
        assert set(row.keys()) == expected_columns


def test_generate_students_values_are_valid_states():
    schema = loader.load_bn_schema()
    node_by_name = {node["name"]: node for node in schema["nodes"]}
    rows = generate_students.generate_students(30, seed=123)
    for row in rows:
        for node_name in generate_students.NODE_ORDER:
            assert row[node_name] in node_by_name[node_name]["states"]


def test_committed_sample_matches_regeneration():
    committed_path = DATA_DIR / "students_seed42.csv"
    with committed_path.open("r", encoding="utf-8", newline="") as handle:
        committed_rows = list(csv.DictReader(handle))

    regenerated = generate_students.generate_students(1000, seed=42)

    assert len(committed_rows) == len(regenerated) == 1000
    for committed, fresh in zip(committed_rows, regenerated):
        assert int(committed["student_id"]) == fresh["student_id"]
        for node_name in generate_students.NODE_ORDER:
            assert committed[node_name] == fresh[node_name]


# ---------------------------------------------------------------------------
# starter loader + student_core stub
# ---------------------------------------------------------------------------


def test_loader_reads_committed_students():
    rows = loader.load_students()
    assert len(rows) == 1000


def test_infer_posterior_is_not_implemented():
    schema = loader.load_bn_schema()
    cpt = loader.load_base_cpt()
    with pytest.raises(NotImplementedError):
        student_core.infer_posterior({}, "Performance", schema, cpt)


def test_starter_does_not_ship_a_working_solver():
    source = (PROJECT_ROOT / "starter" / "student_core.py").read_text(encoding="utf-8")
    assert "raise NotImplementedError" in source
