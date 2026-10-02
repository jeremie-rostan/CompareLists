"""Local name-matching engine.

Pipeline:

1. **Exact pass** - names whose order-insensitive :func:`signature` matches are
   accepted without touching the model.
2. **Blocking** - for each source name, RapidFuzz keeps only the top-K most
   similar target entries, so we never evaluate the full Cartesian product.
3. **Decision** - each surviving pair is forwarded to the local Laya model as a
   three-way ``choice`` question: *does this target entry refer to this source
   person?* (same / different / uncertain). The target entry may contain a parent
   plus one or more children, several names, extra words, or a slight spelling
   variation; the model reads the whole entry. A pair is accepted only when Laya
   answers "same", the probability clears ``accept_threshold`` and the answer is
   sufficiently confident. Borderline pairs are surfaced as *needs review*.

Everything happens on-device: the only network access is the one-time model
download by ``huggingface_hub``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from rapidfuzz import fuzz, process

from .normalize import normalize, signature

ProgressFn = Callable[[int, int, str], None]


@dataclass
class MatchConfig:
    max_candidates: int = 10
    candidate_floor: float = 0.0
    accept_threshold: float = 0.5
    confidence_floor: float = 0.4
    review_threshold: float = 0.4
    auto_accept_ratio: float = 97.0
    use_model: bool = True
    exhaustive: bool = False
    strip_org_suffixes: bool = False
    model_weight: float = 0.5
    subset_match: bool = True


@dataclass
class PairResult:
    source: str
    source_row: int
    target: str
    target_row: int
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
            "matched_entry": self.target,
            "target_row": self.target_row,
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
    def _alpha_key(pair: PairResult):
        return (normalize(pair.source), normalize(pair.target), pair.source_row, pair.target_row)

    def matched_pairs(self) -> list[PairResult]:
        return sorted((p for p in self.pairs if p.decision), key=self._alpha_key)

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
            (p for p in self.pairs if p.needs_review and p.source_row in unmatched_rows),
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


def _contained_targets(source_norm: str, target_norms: list[str]) -> list[int]:
    """Targets that contain the source name's tokens (or vice versa).

    Handles reversed order, extra parent/sibling names, extra or omitted
    surnames and middle names deterministically, without the model. Requires at
    least two tokens on both sides so a lone common first name cannot match.
    """
    source_tokens = set(source_norm.split())
    if len(source_tokens) < 2:
        return []
    hits: list[int] = []
    for idx, target_norm in enumerate(target_norms):
        target_tokens = set(target_norm.split())
        if len(target_tokens) < 2:
            continue
        if source_tokens <= target_tokens or target_tokens <= source_tokens:
            hits.append(idx)
    return hits


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

    source_norm = [normalize(n, strip_org_suffixes=cfg.strip_org_suffixes) for n in source_names]
    target_norm = [normalize(n, strip_org_suffixes=cfg.strip_org_suffixes) for n in target_names]
    source_sig = [signature(n, strip_org_suffixes=cfg.strip_org_suffixes) for n in source_names]
    target_sig = [signature(n, strip_org_suffixes=cfg.strip_org_suffixes) for n in target_names]

    target_by_sig: dict[str, list[int]] = {}
    for idx, sig in enumerate(target_sig):
        target_by_sig.setdefault(sig, []).append(idx)
    unique_target_sigs = sorted(target_by_sig)

    if scorer is not None:
        scorer.ensure_loaded()
        if cfg.use_model:
            scorer.warmup()

    total = len(source_names)
    for i, (src_name, sig) in enumerate(zip(source_names, source_sig)):
        if progress:
            progress(i, total, src_name)

        exact_hits = set(target_by_sig.get(sig) or ())
        subset_hits = (
            set(_contained_targets(source_norm[i], target_norm)) if cfg.subset_match else set()
        )
        deterministic = sorted(exact_hits | subset_hits)
        if deterministic:
            for j in deterministic:
                is_exact = j in exact_hits
                report.pairs.append(
                    PairResult(
                        source=src_name,
                        source_row=i,
                        target=target_names[j],
                        target_row=j,
                        method="exact" if is_exact else "subset",
                        fuzzy=float(fuzz.WRatio(source_norm[i], target_norm[j])),
                        strict=float(fuzz.ratio(sig, target_sig[j])),
                        same_prob=1.0 if is_exact else None,
                        confidence=1.0 if is_exact else None,
                        choice="same" if is_exact else None,
                        decision=True,
                        combined=1.0 if is_exact else 0.99,
                    )
                )
            report.best_by_source[i] = max(report.pairs[-len(deterministic):], key=lambda p: p.combined)
            continue

        candidates = _candidates(
            source_norm[i], sig, unique_target_sigs, target_by_sig, target_norm, target_sig, cfg
        )

        best: Optional[PairResult] = None
        for j, rank, strict in candidates:
            pair = _evaluate_pair(src_name, i, target_names[j], j, rank, strict, scorer, cfg)
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
    unique_target_sigs: list[str],
    target_by_sig: dict[str, list[int]],
    target_norm: list[str],
    target_sig: list[str],
    cfg: MatchConfig,
) -> list[tuple[int, float, float]]:
    if not unique_target_sigs:
        return []

    scored: list[tuple[int, float]] = []
    if cfg.exhaustive:
        for idx, sig in enumerate(target_sig):
            rank = max(fuzz.WRatio(source_norm, target_norm[idx]), fuzz.WRatio(source_sig, sig))
            scored.append((idx, float(rank)))
    else:
        matches = process.extract(
            source_norm, unique_target_sigs, scorer=fuzz.WRatio, limit=cfg.max_candidates
        )
        for sig, score, _ in matches:
            for idx in target_by_sig[sig]:
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

    return [(idx, rank, fuzz.ratio(source_sig, target_sig[idx])) for idx, rank in deduped]


def _evaluate_pair(
    source: str,
    source_row: int,
    target: str,
    target_row: int,
    fuzzy: float,
    strict: float,
    scorer,
    cfg: MatchConfig,
) -> PairResult:
    if strict >= cfg.auto_accept_ratio:
        return PairResult(
            source=source,
            source_row=source_row,
            target=target,
            target_row=target_row,
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
        result = scorer.score(source, target)
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
        target=target,
        target_row=target_row,
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
