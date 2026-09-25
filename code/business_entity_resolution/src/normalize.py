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


# --- Country-conditional structured extraction ---------------------------------
# These return a real value where the country has a recognizable pattern (US,
# India) and None otherwise -- including the unseen France records in the test
# set, so the pipeline degrades gracefully rather than breaking on an
# unrecognized country.

_US_ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_IN_PIN_RE = re.compile(r"\b(\d{6})\b")

_US_STATE_ABBRS = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il",
    "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt",
    "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri",
    "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy",
}


def extract_postal_code(address: str, country: str) -> str | None:
    """Extract a postal/PIN code if the country has a recognizable pattern.

    Returns None for unrecognized countries (e.g. France in the test set) rather
    than raising or guessing -- callers must treat None as "no signal available",
    not as a mismatch.
    """
    if not isinstance(address, str):
        return None
    country_key = str(country).strip().lower()
    if country_key in ("us", "usa", "united states"):
        match = _US_ZIP_RE.search(address)
        return match.group(1) if match else None
    if country_key in ("india", "in"):
        match = _IN_PIN_RE.search(address)
        return match.group(1) if match else None
    return None


def extract_us_state(address: str, country: str) -> str | None:
    """Extract a US state abbreviation token if present; None otherwise."""
    if not isinstance(address, str):
        return None
    if str(country).strip().lower() not in ("us", "usa", "united states"):
        return None
    for token in _base_clean(address).split():
        if token in _US_STATE_ABBRS:
            return token
    return None