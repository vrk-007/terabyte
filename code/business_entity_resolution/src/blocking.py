"""Stage 1: candidate generation (blocking).

Goal here is RECALL, not precision -- the matching model in Stage 3 is responsible
for narrowing candidates down. Measure blocking recall against ground truth on a
held-out train split before trusting this stage; a low recall ceiling here caps
your final F0.5 no matter how good the classifier is.

Strategy: build an inverted index from normalized name tokens (and, separately,
address tokens) to entity_ids, then for each Source 1 entity take the union of all
Source 2 / Source 3 records that share at least one token in either index. This is
plain token overlap -- no external services, no country-specific logic.
"""
from collections import defaultdict

import pandas as pd

from normalize import name_tokens, address_tokens


def _build_inverted_index(df: pd.DataFrame, id_col: str, tokenizer) -> dict:
    index = defaultdict(set)
    for entity_id, text in zip(df[id_col], df["business_name"]):
        for token in tokenizer(text):
            index[token].add(entity_id)
    return index


def _build_address_index(df: pd.DataFrame, id_col: str) -> dict:
    index = defaultdict(set)
    for entity_id, text in zip(df[id_col], df["business_address"]):
        for token in address_tokens(text):
            index[token].add(entity_id)
    return index


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    min_token_len: int = 3,
) -> pd.DataFrame:
    """Return a DataFrame with one row per Source 1 entity and its candidate ids.

    Columns: source1_entity_id, candidate_entity_ids (comma-separated string).
    An entity with no candidates gets an empty string, matching the required
    output format for candidate_pairs.tsv.
    """
    other_df = pd.concat([s2_df, s3_df], ignore_index=True)

    name_index = defaultdict(set)
    for entity_id, text in zip(other_df["entity_id"], other_df["business_name"]):
        for token in name_tokens(text):
            if len(token) >= min_token_len:
                name_index[token].add(entity_id)

    addr_index = _build_address_index(other_df, "entity_id")

    rows = []
    for s1_id, s1_name, s1_addr in zip(
        s1_df["entity_id"], s1_df["business_name"], s1_df["business_address"]
    ):
        candidates: set = set()
        for token in name_tokens(s1_name):
            if len(token) >= min_token_len:
                candidates.update(name_index.get(token, set()))
        for token in address_tokens(s1_addr):
            candidates.update(addr_index.get(token, set()))

        rows.append(
            {
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(sorted(candidates)),
            }
        )

    return pd.DataFrame(rows)


def measure_blocking_recall(
    candidate_df: pd.DataFrame, ground_truth_df: pd.DataFrame
) -> float:
    """Fraction of true (S1, matched_id) pairs that survived blocking.

    Run this on a held-out TRAIN split (where ground truth exists) to sanity-check
    Stage 1 before building Stage 2/3 on top of it.
    """
    candidates_by_s1 = {
        row.source1_entity_id: set(row.candidate_entity_ids.split(","))
        if row.candidate_entity_ids
        else set()
        for row in candidate_df.itertuples()
    }

    total_true, recovered = 0, 0
    for row in ground_truth_df.itertuples():
        true_ids = (
            set(row.matched_entity_ids.split(","))
            if isinstance(row.matched_entity_ids, str) and row.matched_entity_ids
            else set()
        )
        total_true += len(true_ids)
        recovered += len(true_ids & candidates_by_s1.get(row.source1_entity_id, set()))

    return recovered / total_true if total_true else 1.0
