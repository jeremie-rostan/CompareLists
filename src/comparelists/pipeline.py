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

from .normalize import canonicalize_comment, name_parts, normalize, signature

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
    partial_match: bool = True


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
    targets: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

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

    def unmatched_with_candidates(self) -> list[dict]:
        """Unmatched source names, each with its highest-scoring candidate pair.

        Useful as a human review queue: even when nothing cleared the threshold,
        the closest candidate and its scores are shown.
        """
        best_by_source: dict[int, PairResult] = {}
        for pair in self.pairs:
            current = best_by_source.get(pair.source_row)
            if current is None or (pair.combined, pair.same_prob or 0.0) > (
                current.combined,
                current.same_prob or 0.0,
            ):
                best_by_source[pair.source_row] = pair

        rows: list[dict] = []
        for row, name in self.unmatched_sorted():
            entry: dict = {"source_name": name, "source_row": row}
            candidate = best_by_source.get(row)
            if candidate is not None:
                entry.update(
                    {
                        "closest_candidate": candidate.target,
                        "closest_target_row": candidate.target_row,
                        "closest_method": candidate.method,
                        "closest_fuzzy_score": round(candidate.fuzzy, 1),
                        "closest_strict_score": round(candidate.strict, 1),
                        "closest_laya_same_prob": (
                            None if candidate.same_prob is None else round(candidate.same_prob, 4)
                        ),
                        "closest_laya_confidence": (
                            None if candidate.confidence is None else round(candidate.confidence, 4)
                        ),
                        "closest_combined_score": round(candidate.combined, 4),
                    }
                )
            rows.append(entry)
        return rows

    def review_pairs(self) -> list[PairResult]:
        """Borderline pairs belonging to source names that were not matched."""
        unmatched_rows = {row for row, _ in self.unmatched}
        return sorted(
            (p for p in self.pairs if p.needs_review and p.source_row in unmatched_rows),
            key=self._alpha_key,
        )

    def _best_source_per_target(self) -> dict[int, PairResult]:
        best: dict[int, PairResult] = {}
        for pair in self.pairs:
            current = best.get(pair.target_row)
            if current is None or (pair.combined, pair.same_prob or 0.0) > (
                current.combined,
                current.same_prob or 0.0,
            ):
                best[pair.target_row] = pair
        return best

    def _fuzzy_proposal(self, entry: str):
        """Best source for a target that no source ever considered (fuzzy only)."""
        if not self.sources:
            return None
        entry_norm = normalize(entry)
        best = None
        for idx, source in enumerate(self.sources):
            score = float(fuzz.WRatio(normalize(source), entry_norm))
            if best is None or score > best[2]:
                best = (source, idx, score)
        return best

    def target_rows(self) -> list[dict]:
        """Every target entry with its best source proposal (target-centric)."""
        best = self._best_source_per_target()
        accepted_targets = {p.target_row for p in self.pairs if p.decision}
        rows: list[dict] = []
        for idx, entry in enumerate(self.targets):
            row: dict = {
                "target_entry": entry,
                "target_row": idx,
                "used": idx in accepted_targets,
            }
            candidate = best.get(idx)
            if candidate is not None:
                row.update(
                    {
                        "proposed_source": candidate.source,
                        "proposed_source_row": candidate.source_row,
                        "decision": "match" if candidate.decision else "no match",
                        "method": candidate.method,
                        "fuzzy_score": round(candidate.fuzzy, 1),
                        "strict_score": round(candidate.strict, 1),
                        "laya_same_prob": (
                            None if candidate.same_prob is None else round(candidate.same_prob, 4)
                        ),
                        "laya_confidence": (
                            None if candidate.confidence is None else round(candidate.confidence, 4)
                        ),
                        "combined_score": round(candidate.combined, 4),
                    }
                )
            else:
                fallback = self._fuzzy_proposal(entry)
                if fallback is not None:
                    source, source_row, score = fallback
                    row.update(
                        {
                            "proposed_source": source,
                            "proposed_source_row": source_row,
                            "decision": "no match",
                            "method": "fuzzy-proposal",
                            "fuzzy_score": round(score, 1),
                            "strict_score": None,
                            "laya_same_prob": None,
                            "laya_confidence": None,
                            "combined_score": round(score / 100.0, 4),
                        }
                    )
                else:
                    row.update(
                        {
                            "proposed_source": None,
                            "proposed_source_row": None,
                            "decision": "no match",
                            "method": None,
                            "fuzzy_score": None,
                            "strict_score": None,
                            "laya_same_prob": None,
                            "laya_confidence": None,
                            "combined_score": None,
                        }
                    )
            rows.append(row)
        return sorted(rows, key=lambda r: (normalize(r["target_entry"]), r["target_row"]))

    def unmatched_target_rows(self) -> list[dict]:
        """Target entries (receipts) that no source matched, with best proposal."""
        return [row for row in self.target_rows() if not row["used"]]

    def summary(self) -> dict:
        matched_sources = {p.source_row for p in self.matched_pairs()}
        used_targets = {p.target_row for p in self.pairs if p.decision}
        return {
            "source_count": len(self.best_by_source) + len(self.unmatched),
            "matched_source_count": len(matched_sources),
            "unmatched_source_count": len(self.unmatched),
            "target_count": len(self.targets),
            "matched_target_count": len(used_targets),
            "unmatched_target_count": len(self.targets) - len(used_targets),
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


def _partial_targets(source_name: str, target_names: list[str]) -> list[int]:
    """Targets whose child segment names a subset of the source student.

    Handles comments that give the given name plus only *some* surnames, e.g.
    roster ``"Quintana Ibarra, Camila"`` vs comment ``"... -Camila Quintana"``.
    Requires the given name and at least one surname to be present, so siblings
    (different given name) and different families do not match.
    """
    parts = name_parts(source_name)
    if parts is None:
        return []
    surnames, given = parts
    hits: list[int] = []
    for idx, cell in enumerate(target_names):
        comment = canonicalize_comment(cell)
        if " -" not in comment:
            continue
        child_tokens = set(normalize(comment.split(" -")[-1]).split())
        if given <= child_tokens and (surnames & child_tokens):
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
    report = MatchReport(targets=list(target_names), sources=list(source_names))

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
        partial_hits = (
            set(_partial_targets(src_name, target_names)) - exact_hits - subset_hits
            if cfg.partial_match
            else set()
        )
        deterministic = sorted(exact_hits | subset_hits | partial_hits)
        if deterministic:
            for j in deterministic:
                if j in exact_hits:
                    method, combined = "exact", 1.0
                elif j in subset_hits:
                    method, combined = "subset", 0.99
                else:
                    method, combined = "partial-name", 0.95
                is_exact = j in exact_hits
                report.pairs.append(
                    PairResult(
                        source=src_name,
                        source_row=i,
                        target=target_names[j],
                        target_row=j,
                        method=method,
                        fuzzy=float(fuzz.WRatio(source_norm[i], target_norm[j])),
                        strict=float(fuzz.ratio(sig, target_sig[j])),
                        same_prob=1.0 if is_exact else None,
                        confidence=1.0 if is_exact else None,
                        choice="same" if is_exact else None,
                        decision=True,
                        combined=combined,
                    )
                )
            report.best_by_source[i] = max(report.pairs[-len(deterministic):], key=lambda p: p.combined)
            continue

        candidates = _candidates(
            source_norm[i], sig, unique_target_sigs, target_by_sig, target_norm, target_sig, cfg
        )

        model_results = _batch_model_results(scorer, cfg, src_name, target_names, candidates)

        best: Optional[PairResult] = None
        for k, (j, rank, strict) in enumerate(candidates):
            model_result = model_results[k] if model_results is not None else None
            pair = _evaluate_pair(
                src_name, i, target_names[j], j, rank, strict, scorer, cfg, model_result
            )
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


def match_lists_cascade(
    source_names: list[str],
    target_names: list[str],
    primary,
    secondary,
    primary_config: Optional[MatchConfig] = None,
    secondary_config: Optional[MatchConfig] = None,
    progress: Optional[ProgressFn] = None,
    on_secondary_error: Optional[Callable[[str], None]] = None,
) -> MatchReport:
    """Run *primary* first, then *secondary* only on the unmatched source names.

    Used to get Laya's speed for the easy majority and an LLM's reasoning for
    the hard remainder. If the secondary scorer fails (e.g. the local server is
    down), the primary results are returned unchanged and ``on_secondary_error``
    is called with the message.
    """
    report = match_lists(source_names, target_names, scorer=primary, config=primary_config, progress=progress)
    if not report.unmatched or secondary is None:
        return report

    unmatched_names = [name for _, name in report.unmatched]
    original_rows = [row for row, _ in report.unmatched]

    try:
        secondary_report = match_lists(
            unmatched_names, target_names, scorer=secondary, config=secondary_config
        )
    except Exception as exc:  # local server down, bad model, etc.
        if on_secondary_error:
            on_secondary_error(str(exc))
        return report

    for pair in secondary_report.pairs:
        pair.source_row = original_rows[pair.source_row]
    report.pairs.extend(secondary_report.pairs)
    for local_row, pair in secondary_report.best_by_source.items():
        pair.source_row = original_rows[local_row]
        report.best_by_source[pair.source_row] = pair
    report.unmatched = [
        (original_rows[local_row], name) for local_row, name in secondary_report.unmatched
    ]
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


def _batch_model_results(scorer, cfg, source, target_names, candidates):
    """One model call per source when the scorer supports it (e.g. LLM engines)."""
    if not cfg.use_model or scorer is None or not candidates:
        return None
    if not hasattr(scorer, "score_many"):
        return None
    try:
        return scorer.score_many(source, [target_names[j] for j, _, _ in candidates])
    except Exception:
        return None


def _evaluate_pair(
    source: str,
    source_row: int,
    target: str,
    target_row: int,
    fuzzy: float,
    strict: float,
    scorer,
    cfg: MatchConfig,
    model_result: Optional[dict] = None,
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

    if model_result is not None:
        same_prob = model_result.get("same_prob")
        confidence = model_result.get("confidence")
        choice = model_result.get("choice")
    elif cfg.use_model and scorer is not None:
        result = scorer.score(source, target)
        same_prob = result["same_prob"]
        confidence = result["confidence"]
        choice = result["choice"]

    if same_prob is not None:
        combined = (1 - cfg.model_weight) * (fuzzy / 100.0) + cfg.model_weight * same_prob

    if same_prob is None:
        decision = False
    else:
        decision = same_prob >= cfg.accept_threshold and (confidence or 0.0) >= cfg.confidence_floor

    needs_review = not decision and same_prob is not None and same_prob >= cfg.review_threshold

    return PairResult(
        source=source,
        source_row=source_row,
        target=target,
        target_row=target_row,
        method="fuzzy" if same_prob is None else ("llm" if model_result is not None else "laya"),
        fuzzy=fuzzy,
        strict=strict,
        same_prob=same_prob,
        confidence=confidence,
        choice=choice,
        decision=decision,
        combined=combined,
        needs_review=needs_review,
    )
