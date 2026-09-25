# Amazon ML Challenge 2026 — Business Entity Resolution

Read this file in full before writing or modifying any code in this repo. It is the
single source of truth for the task, constraints, and architecture. Do not infer
requirements from elsewhere — if something here is ambiguous, ask rather than guess,
since a format violation causes automatic rejection.

## Problem

Given business records from 3 independent, noisy sources, find all matching records
in Source 2 and Source 3 for each Source 1 entity. Source 1 is the deduplicated
reference set. A Source 1 entity may match zero, one, or many records in Source 2
and/or Source 3.

Fields per record: `entity_id` (prefixed S1-/S2-/S3-), `business_name`,
`business_address`, `country`. Training countries: US, India. **Test set adds
France, unseen in training** — country must be treated as an open string label,
never hardcoded to {US, India}.

Noise to expect: name abbreviations/legal-suffix variants (Corp/Corporation,
Pvt/Private, Ltd/Limited), punctuation differences (& vs "and"), word-order swaps,
typos, transliteration variants; address abbreviations (Rd/Road, St/Street), missing
components, landmark references, reordered components.

## Files

All data files are `.tsv` — always read with `sep="\t"`. A plain `.tsv` read without
this silently collapses into one column.

- `dataset/train/train_source{1,2,3}.tsv`, `dataset/train/train_ground_truth.tsv`
  (ground truth: `source1_entity_id`, `matched_entity_ids` comma-separated, empty
  when no match)
- `dataset/test/test_source{1,2,3}.tsv` (no ground truth)

## Output contract (both required, both tab-separated)

**`output/matching_results.tsv`** — scored on the leaderboard.
Columns: `source1_entity_id`, `matched_entity_ids` (comma-separated S2-/S3- IDs, no
quoting, no duplicates, empty string for singletons).

**`output/candidate_pairs.tsv`** — not scored, used to audit blocking quality.
Columns: `source1_entity_id`, `candidate_entity_ids`. Must be the *final* candidate
set actually fed to the matching model at inference time (last blocking/filtering
stage, not an early pass). Every ID in `matching_results.tsv` must appear here for
the same `source1_entity_id`.

Hard rules (violating any of these causes rejection, not just a low score):
1. Exactly one row per Source 1 **test** entity in both files — every test entity
   must appear, including France entities.
2. No self-matches to Source 1, no IDs outside the test set.
3. No duplicate entity IDs within one list; no duplicate `source1_entity_id` rows.
4. Final model must be MIT/Apache-2.0 licensed and ≤8B parameters.
5. **No external data/API lookups of any kind** (no geocoding, no business-registry
   APIs, no internet augmentation). Only the provided fields may be used.

Validate before every leaderboard upload:
```
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Scoring

Macro-averaged F0.5 per Source 1 entity (precision weighted 2x over recall).
A correctly-predicted singleton (empty list, true empty) scores 1.0; a false merge
on a true singleton scores 0.0. **Bias every design decision toward precision** —
it is better to miss a match than to add a wrong one.

```
F_0.5 = (1.25 × P × R) / (0.25 × P + R)
```

Always validate locally on a held-out slice of `train` (with ground truth) before
spending a leaderboard submission (5/day max).

## Architecture

1. **Blocking / candidate generation** — token or TF-IDF/character-n-gram based
   inverted index or nearest-neighbor search over normalized `business_name` (+
   secondary blocking on address tokens). Optimize this stage for **recall**, not
   precision — measure blocking recall against ground truth explicitly. This
   produces `candidate_pairs.tsv`.
2. **Pairwise feature engineering** — string similarity (Levenshtein/Jaro-Winkler
   ratio, token Jaccard, character n-gram Jaccard, TF-IDF cosine, token-sort ratio)
   on name and address; length/structure features; country match flag (binary,
   never a hardcoded category lookup).
3. **Matching model** — binary classifier (LightGBM/XGBoost) on the pairwise
   features, trained against `train_ground_truth.tsv` labels. An optional small
   Apache/MIT sentence-embedding model (e.g., MiniLM-class, well under 8B params)
   can contribute a semantic-similarity feature.
4. **Threshold tuning** — sweep the classifier's decision threshold on a held-out
   validation split and pick the value that **maximizes macro F0.5 directly**, not
   accuracy/F1/a default 0.5 cutoff.
5. **Output generation** — apply blocking + model + threshold to the full test set,
   write both output files, validate locally.

## Repo layout (this is also the final submission zip layout for `code/` and `output/`)

```
code/business_entity_resolution/
├── src/
│   ├── normalize.py       # name/address text normalization
│   ├── blocking.py        # candidate generation (Stage 1)
│   ├── features.py        # pairwise similarity features (Stage 2)
│   ├── train_model.py     # trains + saves the matcher (Stage 3)
│   ├── predict.py         # end-to-end: test data -> output/*.tsv
│   └── local_eval.py      # held-out F0.5 scoring for iteration
├── models/                # saved trained model artifact(s)
├── README.md              # exact reproduction instructions
└── requirements.txt
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

`notebooks/` (EDA, blocking-recall checks) is scratch space and is **not** part of
the final submission zip — only `code/`, `output/`, and the filled-in
`Documentation_template.md` are.

## Day-by-day plan

- **Day 1**: EDA, normalization utilities, blocking v1 + recall check, trivial
  baseline (normalized exact-name match) submitted once to confirm format.
- **Day 2**: full feature set, train GBM matcher, tune threshold for F0.5 on
  holdout, iterate on blocking if recall ceiling is too low.
- **Day 3**: optional embedding feature, final threshold calibration, full-test-set
  output generation, local validation, `Documentation_template.md` write-up,
  package the final zip.
