"""Name normalisation helpers used for candidate blocking.

The Laya model does the actual semantic decision, but we still need a cheap,
deterministic pre-pass so we never score the full Cartesian product of two
lists. Normalisation is intentionally aggressive: it removes accents,
punctuation and casing, and a *signature* form sorts the tokens so that
"Johnson, Katherine" and "Katherine Johnson" collapse to the same key.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT_RE = re.compile(r"[^0-9a-z\u00c0-\uffff]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Letters NFKD does not fold to ASCII. Folding them helps candidate blocking
# catch transliterated spellings (e.g. "Þórsson" vs "Thorsson").
_TRANSLITERATION = str.maketrans(
    {
        "þ": "th", "ð": "d", "æ": "ae", "ø": "o", "å": "a", "ß": "ss",
        "đ": "d", "ł": "l", "ħ": "h", "ı": "i", "ŧ": "t",
        "Þ": "th", "Ð": "d", "Æ": "ae", "Ø": "o", "Å": "A", "Ł": "L",
    }
)

# Common organisation suffixes that rarely change identity. Used only by the
# optional ``strip_org_suffixes`` normalisation, never by the raw signature.
ORG_SUFFIXES = {
    "corp", "corporation", "inc", "incorporated", "ltd", "limited", "llc",
    "plc", "gmbh", "sa", "sas", "sl", "bv", "nv", "co", "company", "group",
    "holdings", "holding", "international", "intl",
}


def strip_accents(text: str) -> str:
    """Return *text* without diacritics (e.g. ``García`` -> ``Garcia``)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(text: str, *, strip_org_suffixes: bool = False) -> str:
    """Lowercase, de-accent, de-punctuate and collapse whitespace."""
    if text is None:
        return ""
    value = strip_accents(str(text)).translate(_TRANSLITERATION).lower()
    value = _PUNCT_RE.sub(" ", value)
    tokens = value.split()
    if strip_org_suffixes:
        trimmed = [t for t in tokens if t not in ORG_SUFFIXES]
        tokens = trimmed or tokens
    return " ".join(tokens)


def tokens(text: str, *, strip_org_suffixes: bool = False) -> list[str]:
    """Return the normalised tokens of *text*."""
    normalised = normalize(text, strip_org_suffixes=strip_org_suffixes)
    return normalised.split() if normalised else []


def signature(text: str, *, strip_org_suffixes: bool = False) -> str:
    """Order-insensitive key: the sorted, normalised tokens joined by space.

    Two names with the same signature are treated as an exact match without
    ever invoking the model.
    """
    return " ".join(sorted(tokens(text, strip_org_suffixes=strip_org_suffixes)))
