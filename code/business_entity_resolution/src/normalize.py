"""Text normalization for business names and addresses.

Country-agnostic by design: relies on general string cleanup and a small set of
legal-suffix abbreviation expansions, not on country-specific rules, so it degrades
gracefully on the unseen France records in the test set.
"""
import re

# Common legal-suffix / connector abbreviations seen in business names.
# Extend this as EDA turns up more patterns in the actual data.
_NAME_REPLACEMENTS = [
    (r"\bcorp\b\.?", "corporation"),
    (r"\bpvt\b\.?", "private"),
    (r"\bltd\b\.?", "limited"),
    (r"\binc\b\.?", "incorporated"),
    (r"\bco\b\.?", "company"),
    (r"&", " and "),
]

# Common street-type abbreviations for address normalization.
_ADDRESS_REPLACEMENTS = [
    (r"\brd\b\.?", "road"),
    (r"\bst\b\.?", "street"),
    (r"\bave\b\.?", "avenue"),
    (r"\bblvd\b\.?", "boulevard"),
    (r"\bapt\b\.?", "apartment"),
]

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _base_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, expand common legal-suffix abbreviations."""
    text = _base_clean(name)
    for pattern, replacement in _NAME_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return _WS_RE.sub(" ", text).strip()


def normalize_address(address: str) -> str:
    """Lowercase, strip punctuation, expand common street-type abbreviations."""
    text = _base_clean(address)
    for pattern, replacement in _ADDRESS_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return _WS_RE.sub(" ", text).strip()


def name_tokens(name: str) -> set:
    """Token set for a normalized name, useful for Jaccard/blocking."""
    return set(normalize_name(name).split())


def address_tokens(address: str) -> set:
    """Token set for a normalized address, useful for Jaccard/blocking."""
    return set(normalize_address(address).split())
