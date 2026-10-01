"""Expand a messy target cell into candidate name variants.

Real rosters often pack several people into one cell::

    "María González Pedro Ramírez"      # parent first, then the student
    "Juan Pérez y Lucía Pérez"          # siblings joined by a conjunction
    "Kathy and Erica Thompson"          # two students

We do not know which token belongs to whom, so we generate every plausible
name:

* ``spans`` - every contiguous run of 1..``max_tokens`` tokens (catches a name
  that sits at the start, middle or end of the cell). The full cell is always
  included, so a clean single-name column still behaves exactly as before.
* ``conjunctions`` - split on ``y``/``and``/``&``/``,``/... and keep each part.
* ``none`` - leave the cell untouched.

Variants keep their original accents and casing (Laya reads them better); the
pipeline normalises them separately for the cheap exact/blocking pass.
"""

from __future__ import annotations

import re

from .normalize import normalize

CONJUNCTIONS = {
    "y", "e", "and", "&", "+", "/", "o", "u", "or",
    "tambien", "también", "also", "et", "und",
}

_SEPARATOR_RE = re.compile(r"[,/&+;|]+")
_MAX_VARIANTS = 40


def _raw_tokens(text: str) -> list[str]:
    """Split on punctuation but keep accents and case."""
    if not text:
        return []
    cleaned = _SEPARATOR_RE.sub(" ", str(text))
    return [tok for tok in cleaned.split() if tok]


def _is_conjunction(token: str) -> bool:
    return token.lower() in CONJUNCTIONS


def _keep(span: list[str]) -> bool:
    joined = normalize(" ".join(span))
    if len(joined) < 2:
        return False
    if all(_is_conjunction(tok) for tok in span):
        return False
    return True


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = normalize(value)
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def expand_variants(text: str, mode: str = "spans", max_tokens: int = 4) -> list[str]:
    """Return candidate names contained in *text*."""
    tokens = _raw_tokens(text)
    if not tokens:
        return []

    if mode == "none" or len(tokens) == 1:
        return _dedupe([" ".join(tokens)])[:_MAX_VARIANTS]

    variants: list[str] = []

    if mode == "conjunctions":
        segment: list[str] = []
        for token in tokens:
            if _is_conjunction(token):
                if segment:
                    variants.append(" ".join(segment))
                    segment = []
            else:
                segment.append(token)
        if segment:
            variants.append(" ".join(segment))
        variants.append(" ".join(tokens))
        return _dedupe(variants)[:_MAX_VARIANTS]

    # default: spans
    limit = max(1, max_tokens)
    n = len(tokens)
    for size in range(1, min(limit, n) + 1):
        for start in range(0, n - size + 1):
            span = tokens[start : start + size]
            if _keep(span):
                variants.append(" ".join(span))
    variants.append(" ".join(tokens))
    return _dedupe(variants)[:_MAX_VARIANTS]
