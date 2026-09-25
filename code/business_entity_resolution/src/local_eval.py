"""Held-out evaluation: macro F0.5, and threshold sweep to maximize it.

TODO (Day 2): split train_source1 entities into train/holdout, run the full
pipeline (blocking -> features -> trained model) on the holdout, compare
predictions to train_ground_truth.tsv, and report macro F0.5 at several decision
thresholds so you can pick the one to use in predict.py.
"""


def f0_5(precision: float, recall: float) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    beta_sq = 0.25
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def score_entity(predicted_ids: set, true_ids: set) -> float:
    """F0.5 for a single Source 1 entity. Singletons (true_ids empty) score 1.0
    for an empty prediction and 0.0 for any false-positive prediction.
    """
    if not true_ids:
        return 1.0 if not predicted_ids else 0.0
    if not predicted_ids:
        return 0.0
    tp = len(predicted_ids & true_ids)
    precision = tp / len(predicted_ids)
    recall = tp / len(true_ids)
    return f0_5(precision, recall)


def macro_f0_5(predictions: dict, ground_truth: dict) -> float:
    """predictions, ground_truth: dict[source1_entity_id -> set[matched_ids]]."""
    scores = [
        score_entity(predictions.get(s1_id, set()), true_ids)
        for s1_id, true_ids in ground_truth.items()
    ]
    return sum(scores) / len(scores) if scores else 0.0


if __name__ == "__main__":
    raise NotImplementedError("Wire up the held-out split + pipeline run here.")
