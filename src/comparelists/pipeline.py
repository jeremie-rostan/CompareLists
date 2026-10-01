"""Local name-matching engine.

Pipeline:

1. **Exact pass** - names whose order-insensitive :func:`signature` matches are
   accepted without touching the model.
2. **Blocking** - for everything else we use RapidFuzz to keep only the top-K
   most similar targets per source name, so we never evaluate the full
   Cartesian product.
3. **Decision** - each surviving pair is forwarded to the local Laya model as a
   three-way ``choice`` question ("same" / "different" / "uncertain"). Laya
   returns calibrated probabilities; a pair is accepted only when Laya says
   "same", the probability clears ``accept_threshold`` and the answer is
   suffiently confident. Borderline pairs are surfaced as *needs review*.

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
    confidence_floor: float = 0.25
    review_threshold: float = 0.4
    auto_accept_ratio: float = 97.0
    use_model: bool = True
    exhaustive: bool = False
    strip_org_suffixes: bool = False
    model_weight: float = 0.5


@dataclass
class PairResult:
    source: str
    source_row: int
    target: str
    target_row: int
    method: str
    fuzzy: float
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
            "matched_name": self.target,
            "matched_row": self.target_row,
            "decision": "match" if self.decision else "no match",
            "needs_review": self.needs_review,
            "method": self.method,
            "laya_choice": self.choice,
            "fuzzy_score": round(self.fuzzy, 1),
            "laya_same_prob": None if self.same_prob is None else round(self.same_prob, 4),
            "laya_confidence": None if self.confidence is None else round(self.confidence, 4),
            "combined_score": round(self.combined, 4),
        }


@dataclass
class MatchReport:
    pairs: list[PairResult] = field(default_factory=list)
    best_by_source: dict[int, PairResult] = field(default_factory=dict)
    unmatched: list[tuple[int, str]] = field(default_factory=list)

    def matched_pairs(self) -> list[PairResult]:
        return [p for p in self.pairs if p.decision]

    def best_matches(self) -> list[PairResult]:
        return sorted(self.best_by_source.values(), key=lambda p: p.source_row)

    def review_pairs(self) -> list[PairResult]:
        """Borderline pairs belonging to source names that were not matched."""
        unmatched_rows = {row for row, _ in self.unmatched}
        return [p for p in self.pairs if p.needs_review and p.source_row in unmatched_rows]

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


def _unique_signatures(names: Iterable[str], strip_org: bool) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for idx, name in enumerate(names):
        mapping.setdefault(signature(name, strip_org_suffixes=strip_org), []).append(idx)
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

    source_norm = [normalize(n, strip_org_suffixes=cfg.strip_org_suffixes) for n in source_names]
    target_norm = [normalize(n, strip_org_suffixes=cfg.strip_org_suffixes) for n in target_names]
    source_sig = [signature(n, strip_org_suffixes=cfg.strip_org_suffixes) for n in source_names]
    target_sig = [signature(n, strip_org_suffixes=cfg.strip_org_suffixes) for n in target_names]

    target_by_sig = _unique_signatures(target_names, cfg.strip_org_suffixes)
    unique_target_sigs = sorted(target_by_sig)

    if scorer is not None:
        scorer.ensure_loaded()
        if cfg.use_model:
            scorer.warmup()

    total = len(source_names)
    for i, (src_name, sig) in enumerate(zip(source_names, source_sig)):
        if progress:
            progress(i, total, src_name)

        exact_targets = target_by_sig.get(sig)
        if exact_targets:
            for j in exact_targets:
                report.pairs.append(
                    PairResult(
                        source=src_name,
                        source_row=i,
                        target=target_names[j],
                        target_row=j,
                        method="exact",
                        fuzzy=100.0,
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
            source_norm[i], unique_target_sigs, target_by_sig, target_sig, target_norm, cfg
        )

        best: Optional[PairResult] = None
        for j, fuzzy in candidates:
            pair = _evaluate_pair(src_name, i, target_names[j], j, fuzzy, scorer, cfg)
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
    unique_target_sigs: list[str],
    target_by_sig: dict[str, list[int]],
    target_sig: list[str],
    target_norm: list[str],
    cfg: MatchConfig,
) -> list[tuple[int, float]]:
    if not unique_target_sigs:
        return []

    if cfg.exhaustive:
        scored = [
            (idx, max(fuzz.WRatio(source_norm, target_norm[idx]), fuzz.WRatio(sig, target_sig[idx])))
            for idx, sig in enumerate(target_sig)
        ]
    else:
        matches = process.extract(
            source_norm, unique_target_sigs, scorer=fuzz.WRatio, limit=cfg.max_candidates
        )
        scored = []
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
    return deduped


def _evaluate_pair(
    source: str,
    source_row: int,
    target: str,
    target_row: int,
    fuzzy: float,
    scorer,
    cfg: MatchConfig,
) -> PairResult:
    same_prob: Optional[float] = None
    confidence: Optional[float] = None
    choice: Optional[str] = None

    if fuzzy >= cfg.auto_accept_ratio:
        return PairResult(
            source=source,
            source_row=source_row,
            target=target,
            target_row=target_row,
            method="fuzzy-high",
            fuzzy=fuzzy,
            same_prob=None,
            confidence=None,
            choice=None,
            decision=True,
            combined=1.0,
        )

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

    needs_review = (
        not decision and same_prob is not None and same_prob >= cfg.review_threshold
    )

    return PairResult(
        source=source,
        source_row=source_row,
        target=target,
        target_row=target_row,
        method="laya" if same_prob is not None else "fuzzy",
        fuzzy=fuzzy,
        same_prob=same_prob,
        confidence=confidence,
        choice=choice,
        decision=decision,
        combined=combined,
        needs_review=needs_review,
    )
