# Business Entity Resolution — Pipeline

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` end-to-end
from the provided train/test data. See `../../PROJECT_BRIEF.md` (repo root) for the
full problem spec, constraints, and architecture this code implements.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

1. Train the matcher (reads `dataset/train/*`, writes `models/matcher.pkl`):
   ```bash
   python3 src/train_model.py
   ```
2. Check local validation F0.5 before touching the leaderboard:
   ```bash
   python3 src/local_eval.py
   ```
3. Generate the full test-set submission:
   ```bash
   python3 src/predict.py
   ```
4. Validate format before uploading:
   ```bash
   python3 ../../utils/validate_submission.py \
     --matching ../../output/matching_results.tsv \
     --candidate ../../output/candidate_pairs.tsv \
     --test-dir ../../dataset/test
   ```

## Module overview

- `normalize.py` — text normalization for business names and addresses.
- `blocking.py` — candidate generation (Stage 1); optimized for recall.
- `features.py` — pairwise similarity feature engineering (Stage 2).
- `train_model.py` — trains the binary match/no-match classifier (Stage 3).
- `local_eval.py` — held-out split + macro F0.5 scoring, used to tune the decision
  threshold before any leaderboard submission.
- `predict.py` — applies blocking + trained model + tuned threshold to the test set
  and writes both output files.
