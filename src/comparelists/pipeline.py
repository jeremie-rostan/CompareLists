"""Local name-matching engine.

Pipeline:

1. **Expand** - each target cell is turned into candidate name variants
   (:mod:`comparelists.variants`), so a cell that holds a parent plus one or more
   students, or siblings joined by "y"/"and", can still match.
2. **Exact pass** - names whose order-insensitive :func:`signature` matches are
   accepted without touching the model.
3. **Blocking** - for everything else we use RapidFuzz to keep only the top-K
   most similar target variants per source name, so we never evaluate the full
   Cartesian product.
4. **Decision** - each surviving pair is forwarded to the local Laya model as a
   three-way ``choice`` question ("same" / "different" / "uncertain"). A pair is
   accepted only when Laya says "same", the probability clears
   ``accept_threshold`` and the answer is sufficiently confident. Borderline
   pairs are surfaced as *needs review*.

Everything happens on-device: the only network access is the one-time model
download by ``huggingface_hub``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from rapidfuzz import fuzz, process

from .normalize import normalize, signature
from .variants import expand_variants

ProgressFn = Callable[[int, int, str], None]


@dataclass
class MatchConfig:
    max_candidates: int = 10
    candidate_floor: float = 0.0
    accept_threshold: float = 0.5
    confidence_floor: float = 0.25
    review_threshold: float = 0.4
    auto_accept_ratio: float = 97.0
    use_model: bool = True
    exhaustive: bool = False
    strip_org_suffixes: bool = False
    model_weight: float = 0.5
    target_format: str = "spans"
    variant_max_tokens: int = 4


@dataclass
class TargetEntry:
    row: int
    cell: str
    variant: str
    norm: str
    sig: str


@dataclass
class PairResult:
    source: str
    source_row: int
    target: str
    target_row: int
    variant: str
    method: str
    fuzzy: float
    strict: float
    same_prob: Optional[float]
    confidence: Optional[float]
    choice: Optional[str]
    decision: bool
    combined: float
    needs_review: bool = False

    def as_row(self) -> dict:
        return {
            "source_name": self.source,
            "source_row": self.source_row,
            "target_cell": self.target,
            "target_row": self.target_row,
            "target_variant": self.variant,
            "decision": "match" if self.decision else "no match",
            "needs_review": self.needs_review,
            "method": self.method,
            "laya_choice": self.choice,
            "fuzzy_score": round(self.fuzzy, 1),
            "strict_score": round(self.strict, 1),
            "laya_same_prob": None if self.same_prob is None else round(self.same_prob, 4),
            "laya_confidence": None if self.confidence is None else round(self.confidence, 4),
            "combined_score": round(self.combined, 4),
        }


@dataclass
class MatchReport:
    pairs: list[PairResult] = field(default_factory=list)
    best_by_source: dict[int, PairResult] = field(default_factory=dict)
    unmatched: list[tuple[int, str]] = field(default_factory=list)

    @staticmethod
    def _best_per_row(pairs: Iterable[PairResult]) -> list[PairResult]:
        best: dict[tuple[int, int], PairResult] = {}
        for pair in pairs:
            key = (pair.source_row, pair.target_row)
            if key not in best or pair.combined > best[key].combined:
                best[key] = pair
        return sorted(best.values(), key=lambda p: (p.source_row, p.target_row))

    @staticmethod
    def _alpha_key(pair: PairResult):
        return (normalize(pair.source), normalize(pair.target), pair.source_row, pair.target_row)

    def matched_pairs(self) -> list[PairResult]:
        return sorted(
            self._best_per_row(p for p in self.pairs if p.decision), key=self._alpha_key
        )

    def best_matches(self) -> list[PairResult]:
        return sorted(self.best_by_source.values(), key=self._alpha_key)

    def all_pairs_sorted(self) -> list[PairResult]:
        return sorted(self.pairs, key=self._alpha_key)

    def unmatched_sorted(self) -> list[tuple[int, str]]:
        return sorted(self.unmatched, key=lambda item: (normalize(item[1]), item[0]))

    def review_pairs(self) -> list[PairResult]:
        """Borderline pairs belonging to source names that were not matched."""
        unmatched_rows = {row for row, _ in self.unmatched}
        return sorted(
            self._best_per_row(
                p for p in self.pairs if p.needs_review and p.source_row in unmatched_rows
            ),
            key=self._alpha_key,
        )

    def summary(self) -> dict:
        matched_sources = {p.source_row for p in self.matched_pairs()}
        return {
            "source_count": len(self.best_by_source) + len(self.unmatched),
            "matched_source_count": len(matched_sources),
            "unmatched_source_count": len(self.unmatched),
            "matched_pair_count": len(self.matched_pairs()),
            "review_count": len(self.review_pairs()),
            "pairs_evaluated": len(self.pairs),
        }


def _build_entries(target_names: list[str], cfg: MatchConfig) -> list[TargetEntry]:
    strip_org = cfg.strip_org_suffixes
    entries: list[TargetEntry] = []
    for row, cell in enumerate(target_names):
        variants = expand_variants(cell, mode=cfg.target_format, max_tokens=cfg.variant_max_tokens)
        if not variants:
            variants = [cell]
        for variant in variants:
            entries.append(
                TargetEntry(
                    row=row,
                    cell=cell,
                    variant=variant,
                    norm=normalize(variant, strip_org_suffixes=strip_org),
                    sig=signature(variant, strip_org_suffixes=strip_org),
                )
            )
    return entries


def _index_by_signature(entries: list[TargetEntry]) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for idx, entry in enumerate(entries):
        mapping.setdefault(entry.sig, []).append(idx)
    return mapping


def match_lists(
    source_names: list[str],
    target_names: list[str],
    scorer=None,
    config: Optional[MatchConfig] = None,
    progress: Optional[ProgressFn] = None,
) -> MatchReport:
    """Match every source name against the target list."""
    cfg = config or MatchConfig()
    report = MatchReport()

    entries = _build_entries(target_names, cfg)
    entry_by_sig = _index_by_signature(entries)
    unique_entry_sigs = sorted(entry_by_sig)

    source_norm = [normalize(n, strip_org_suffixes=cfg.strip_org_suffixes) for n in source_names]
    source_sig = [signature(n, strip_org_suffixes=cfg.strip_org_suffixes) for n in source_names]

    if scorer is not None:
        scorer.ensure_loaded()
        if cfg.use_model:
            scorer.warmup()

    total = len(source_names)
    for i, (src_name, sig) in enumerate(zip(source_names, source_sig)):
        if progress:
            progress(i, total, src_name)

        exact = entry_by_sig.get(sig)
        if exact:
            for entry_idx in exact:
                entry = entries[entry_idx]
                report.pairs.append(
                    PairResult(
                        source=src_name,
                        source_row=i,
                        target=entry.cell,
                        target_row=entry.row,
                        variant=entry.variant,
                        method="exact",
                        fuzzy=100.0,
                        strict=100.0,
                        same_prob=1.0,
                        confidence=1.0,
                        choice="same",
                        decision=True,
                        combined=1.0,
                    )
                )
            report.best_by_source[i] = report.pairs[-1]
            continue

        candidates = _candidates(
            source_norm[i], sig, unique_entry_sigs, entry_by_sig, entries, cfg
        )

        best: Optional[PairResult] = None
        for entry_idx, rank, strict in candidates:
            pair = _evaluate_pair(src_name, i, entries[entry_idx], rank, strict, scorer, cfg)
            report.pairs.append(pair)
            if pair.decision and (best is None or pair.combined > best.combined):
                best = pair

        if best is None:
            report.unmatched.append((i, src_name))
        else:
            report.best_by_source[i] = best

    if progress:
        progress(total, total, "done")
    return report


def _candidates(
    source_norm: str,
    source_sig: str,
    unique_entry_sigs: list[str],
    entry_by_sig: dict[str, list[int]],
    entries: list[TargetEntry],
    cfg: MatchConfig,
) -> list[tuple[int, float, float]]:
    if not unique_entry_sigs:
        return []

    scored: list[tuple[int, float]] = []
    if cfg.exhaustive:
        for idx, entry in enumerate(entries):
            rank = max(fuzz.WRatio(source_norm, entry.norm), fuzz.WRatio(source_sig, entry.sig))
            scored.append((idx, float(rank)))
    else:
        matches = process.extract(
            source_norm, unique_entry_sigs, scorer=fuzz.WRatio, limit=cfg.max_candidates
        )
        for sig, score, _ in matches:
            for idx in entry_by_sig[sig]:
                scored.append((idx, float(score)))

    if cfg.candidate_floor > 0:
        scored = [(idx, s) for idx, s in scored if s >= cfg.candidate_floor]

    seen: set[int] = set()
    deduped: list[tuple[int, float]] = []
    for idx, score in scored:
        if idx in seen:
            continue
        seen.add(idx)
        deduped.append((idx, score))
    if not cfg.exhaustive:
        deduped.sort(key=lambda x: x[1], reverse=True)
        deduped = deduped[: cfg.max_candidates]

    return [(idx, rank, fuzz.ratio(source_sig, entries[idx].sig)) for idx, rank in deduped]


def _evaluate_pair(
    source: str,
    source_row: int,
    entry: TargetEntry,
    fuzzy: float,
    strict: float,
    scorer,
    cfg: MatchConfig,
) -> PairResult:
    if strict >= cfg.auto_accept_ratio:
        return PairResult(
            source=source,
            source_row=source_row,
            target=entry.cell,
            target_row=entry.row,
            variant=entry.variant,
            method="fuzzy-high",
            fuzzy=fuzzy,
            strict=strict,
            same_prob=None,
            confidence=None,
            choice=None,
            decision=True,
            combined=1.0,
        )

    same_prob: Optional[float] = None
    confidence: Optional[float] = None
    choice: Optional[str] = None
    combined = fuzzy / 100.0

    if cfg.use_model and scorer is not None:
        result = scorer.score(source, entry.variant)
        same_prob = result["same_prob"]
        confidence = result["confidence"]
        choice = result["choice"]
        combined = (1 - cfg.model_weight) * (fuzzy / 100.0) + cfg.model_weight * same_prob

    if same_prob is None:
        decision = False
    else:
        decision = (
            choice == "same"
            and same_prob >= cfg.accept_threshold
            and (confidence or 0.0) >= cfg.confidence_floor
        )

    needs_review = not decision and same_prob is not None and same_prob >= cfg.review_threshold

    return PairResult(
        source=source,
        source_row=source_row,
        target=entry.cell,
        target_row=entry.row,
        variant=entry.variant,
        method="laya" if same_prob is not None else "fuzzy",
        fuzzy=fuzzy,
        strict=strict,
        same_prob=same_prob,
        confidence=confidence,
        choice=choice,
        decision=decision,
        combined=combined,
        needs_review=needs_review,
    )
