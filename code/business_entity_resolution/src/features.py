"""Stage 2: pairwise similarity features for (Source 1, candidate) pairs.

All features are computed purely from the two records' own text fields -- no
external lookups, per the challenge's fair-play rules. Country is used only as a
same/different flag, never as a hardcoded category, so behavior degrades
gracefully on the unseen France test records.
"""
import jellyfish
import pandas as pd
from rapidfuzz import fuzz

from normalize import address_tokens, name_tokens, normalize_address, normalize_name


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_features(
    s1_name: str,
    s1_address: str,
    s1_country: str,
    cand_name: str,
    cand_address: str,
    cand_country: str,
) -> dict:
    """Feature vector for a single (Source 1, candidate) pair."""
    n1, n2 = normalize_name(s1_name), normalize_name(cand_name)
    a1, a2 = normalize_address(s1_address), normalize_address(cand_address)

    return {
        "name_exact_match": float(n1 == n2),
        "name_levenshtein_ratio": fuzz.ratio(n1, n2) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(n1, n2) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(n1, n2) / 100.0,
        "name_jaccard": _jaccard(name_tokens(s1_name), name_tokens(cand_name)),
        "name_jaro_winkler": jellyfish.jaro_winkler_similarity(n1, n2),
        "name_len_diff": abs(len(n1) - len(n2)),
        "address_levenshtein_ratio": fuzz.ratio(a1, a2) / 100.0,
        "address_token_sort_ratio": fuzz.token_sort_ratio(a1, a2) / 100.0,
        "address_jaccard": _jaccard(
            address_tokens(s1_address), address_tokens(cand_address)
        ),
        "country_match": float(str(s1_country) == str(cand_country)),
    }


def build_feature_matrix(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """pairs_df must have columns: s1_name, s1_address, s1_country,
    cand_name, cand_address, cand_country (plus any id columns to carry through).
    Returns a DataFrame of engineered features, row-aligned with pairs_df.
    """
    records = [
        pair_features(
            row.s1_name,
            row.s1_address,
            row.s1_country,
            row.cand_name,
            row.cand_address,
            row.cand_country,
        )
        for row in pairs_df.itertuples()
    ]
    return pd.DataFrame(records)
