"""Stage 3: train the match/no-match classifier.

TODO (Day 2): flesh this out once blocking recall (blocking.py) has been checked
on a held-out train split. Skeleton below shows the intended flow.
"""
import pandas as pd
from lightgbm import LGBMClassifier

from blocking import generate_candidates
from features import build_feature_matrix

DATA_DIR = "../../dataset/train"
MODEL_PATH = "models/matcher.pkl"


def load_train():
    s1 = pd.read_csv(f"{DATA_DIR}/train_source1.tsv", sep="\t")
    s2 = pd.read_csv(f"{DATA_DIR}/train_source2.tsv", sep="\t")
    s3 = pd.read_csv(f"{DATA_DIR}/train_source3.tsv", sep="\t")
    gt = pd.read_csv(f"{DATA_DIR}/train_ground_truth.tsv", sep="\t")
    return s1, s2, s3, gt


def build_labeled_pairs(s1, s2, s3, gt, candidate_df):
    """Join blocking candidates with ground truth to produce labeled pairs.

    TODO: for each (source1_entity_id, candidate_entity_id) in candidate_df,
    label 1 if candidate_entity_id is in the ground-truth matched set for that
    source1_entity_id, else 0. Then attach the raw name/address/country fields
    from s1/s2/s3 (by entity_id) so build_feature_matrix can run on it.
    """
    raise NotImplementedError


def main():
    s1, s2, s3, gt = load_train()
    candidate_df = generate_candidates(s1, s2, s3)
    labeled_pairs = build_labeled_pairs(s1, s2, s3, gt, candidate_df)
    X = build_feature_matrix(labeled_pairs)
    y = labeled_pairs["label"]

    model = LGBMClassifier(n_estimators=300, max_depth=6)
    model.fit(X, y)

    import joblib

    joblib.dump(model, MODEL_PATH)


if __name__ == "__main__":
    main()
