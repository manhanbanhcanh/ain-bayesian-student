# Project 05 data dictionary

**All files in this folder are synthetic educational data.** They describe a fictional Bayesian
Network simulation, not real students, real attendance, or real exam results.

## Files

| File | Contents |
|---|---|
| `bn_schema.json` | Network structure: node names, states, parents, edges, independence notes. |
| `base_cpt.json` | Conditional probability table for every node. Every conditional row sums to 1.0. |
| `scenarios.json` | Four evidence scenarios (little evidence, strong positive, conflicting, CPT sensitivity) plus the required experiment description. |
| `students_seed42.csv` | 1,000 synthetic student rows sampled with seed 42 (the default). |
| `SHA256SUMS` | Checksums of the files above, so regenerated data can be verified byte-for-byte. |

## Columns in `students_seed42.csv`

`student_id, PriorPreparation, Attendance, ExamDifficulty, StudyConsistency, AssignmentCompletion, Performance`

Each row is one synthetic student sampled by ancestral sampling from the network in
`bn_schema.json`, using the conditional probability tables in `base_cpt.json`, in the fixed
topological order: `PriorPreparation`, `Attendance`, `ExamDifficulty`, `StudyConsistency`,
`AssignmentCompletion`, `Performance`.

## Regenerating the data

```bash
python3.11 scripts/generate_students.py --n 1000 --seed 42
```

This overwrites `data/students_seed42.csv` with a byte-identical copy of the committed file
(verify with `sha256sum -c SHA256SUMS` from inside `data/`). A larger, non-committed sample is
optional:

```bash
python3.11 scripts/generate_students.py --n 5000 --seed 42
```

This writes `data/students_n5000_seed42.csv` without touching the committed 1,000-row file.

## Verifying checksums

```bash
cd data
shasum -a 256 -c SHA256SUMS
```
