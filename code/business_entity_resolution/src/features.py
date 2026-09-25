"""Stage 2: pairwise similarity features for (Source 1, candidate) pairs.

All features are computed purely from the two records' own text fields -- no
external lookups, per the challenge's fair-play rules. Country is used only as a
same/different flag or a conditional structured extractor, never a hardcoded
category, so behavior degrades gracefully on the unseen France test records.
"""
import jellyfish
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from normalize import (
    address_tokens,
    extract_postal_code,
    extract_us_state,
    name_tokens,
    normalize_address,
    normalize_name,
)


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

    # Country-conditional structured signals. None means "no recognized
    # pattern" (e.g. France) -- encoded as neutral 0.5, not a hard mismatch,
    # so the model isn't penalized for unrecognized countries.
    pin1 = extract_postal_code(s1_address, s1_country)
    pin2 = extract_postal_code(cand_address, cand_country)
    postal_match = float(pin1 == pin2) if pin1 is not None and pin2 is not None else 0.5

    state1 = extract_us_state(s1_address, s1_country)
    state2 = extract_us_state(cand_address, cand_country)
    state_match = float(state1 == state2) if state1 is not None and state2 is not None else 0.5

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
        "postal_code_match": postal_match,
        "us_state_match": state_match,
    }


# --- TF-IDF cosine similarity ---------------------------------------------------
# Unlike the features above, this needs corpus-level statistics (term/document
# frequencies), so it can't be computed per-pair in isolation. Fit once on all
# names in the dataset (S1 + S2 + S3, train or test as appropriate), then reuse
# the fitted vectorizer to score every pair.


def fit_name_tfidf_vectorizer(all_names: list) -> TfidfVectorizer:
    """Fit a character n-gram TF-IDF vectorizer over every business name in the
    dataset you're about to score. Character n-grams (not word n-grams) are
    used because they tolerate typos, transliteration variants, and partial
    substring overlap better than word-level TF-IDF.
    """
    normalized = [normalize_name(n) for n in all_names]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    vectorizer.fit(normalized)
    return vectorizer


def name_tfidf_cosine(vectorizer: TfidfVectorizer, name1: str, name2: str) -> float:
    """Cosine similarity between two names' TF-IDF vectors under a pre-fitted
    vectorizer. Returns a value in [0, 1].
    """
    vecs = vectorizer.transform([normalize_name(name1), normalize_name(name2)])
    return float(cosine_similarity(vecs[0], vecs[1])[0, 0])


def build_feature_matrix(
    pairs_df: pd.DataFrame, tfidf_vectorizer: TfidfVectorizer = None
) -> pd.DataFrame:
    """pairs_df must have columns: s1_name, s1_address, s1_country,
    cand_name, cand_address, cand_country (plus any id columns to carry through).

    Pass a vectorizer fitted with fit_name_tfidf_vectorizer() (over the full set
    of names you're scoring) to include the TF-IDF cosine feature. If omitted,
    that column is left out -- useful for quick iteration before you've decided
    on corpus scope (train-only vs. train+test).
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
    feature_df = pd.DataFrame(records)

    if tfidf_vectorizer is not None:
        feature_df["name_tfidf_cosine"] = [
            name_tfidf_cosine(tfidf_vectorizer, row.s1_name, row.cand_name)
            for row in pairs_df.itertuples()
        ]

    return feature_df