"""End-to-end: test data -> output/matching_results.tsv + output/candidate_pairs.tsv.

TODO (Day 3): load the trained model + tuned threshold from local_eval.py, run
blocking + feature extraction + model scoring on the full test set, and write both
required output files in the exact format PROJECT_BRIEF.md specifies.
"""
import pandas as pd

DATA_DIR = "../../dataset/test"
OUTPUT_DIR = "../../output"
MODEL_PATH = "models/matcher.pkl"
DECISION_THRESHOLD = 0.5  # TODO: replace with the value tuned in local_eval.py


def load_test():
    s1 = pd.read_csv(f"{DATA_DIR}/test_source1.tsv", sep="\t")
    s2 = pd.read_csv(f"{DATA_DIR}/test_source2.tsv", sep="\t")
    s3 = pd.read_csv(f"{DATA_DIR}/test_source3.tsv", sep="\t")
    return s1, s2, s3


def main():
    raise NotImplementedError(
        "Wire up: generate_candidates -> build_feature_matrix -> model.predict_proba "
        "-> threshold -> write candidate_pairs.tsv and matching_results.tsv"
    )


if __name__ == "__main__":
    main()
