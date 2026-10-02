"""Reliability harness for CompareLists.

Runs the real engine over a small labelled set of name pairs and reports false
positives / false negatives, so wording, model and threshold changes can be
checked instead of guessed.

    uv run python evaluation/eval_matching.py [model_id]

Defaults to the fine-tuned ``jeremierostan/laya-pairmatch`` checkpoint. This
labelled set is deliberately out-of-distribution for that model (bare name
pairs, no parent comment), so it is a conservative probe of the hardest cases.
"""

from __future__ import annotations

import csv
import os
import sys

os.environ.setdefault("USE_TF", "0")

from comparelists.laya_matcher import LayaNameScorer
from comparelists.pipeline import MatchConfig, match_lists

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "labeled_pairs.csv")


def load_pairs():
    with open(DATASET, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [(r["source"], r["target"], int(r["expected"])) for r in rows]


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "jeremierostan/laya-pairmatch"
    pairs = load_pairs()
    sources = list(dict.fromkeys(s for s, _, _ in pairs))
    targets = list(dict.fromkeys(t for _, t, _ in pairs))
    expected = {(s, t) for s, t, want in pairs if want}

    scorer = LayaNameScorer(model_id=model_id)
    config = MatchConfig(
        exhaustive=True,
        accept_threshold=scorer.config["default_accept_threshold"],
        confidence_floor=scorer.config["default_confidence_floor"],
    )
    report = match_lists(sources, targets, scorer=scorer, config=config)
    accepted = {(p.source, p.target) for p in report.matched_pairs()}
    review = {(p.source, p.target) for p in report.review_pairs()}

    fp = sorted(accepted - expected)
    fn = sorted(expected - accepted)
    tp = len(expected & accepted)
    precision = tp / max(tp + len(fp), 1)
    recall = tp / max(len(expected), 1)

    print(f"model:              {model_id} (preset: {scorer.preset})")
    print(f"labelled positives: {len(expected)}")
    print(f"true positives:     {tp}")
    print(f"false positives:    {len(fp)}")
    print(f"false negatives:    {len(fn)}")
    print(f"precision:          {precision:.3f}")
    print(f"recall:             {recall:.3f}")
    if fp:
        print("\nfalse positives:")
        for s, t in fp:
            print(f"  {s!r} -> {t!r}")
    if fn:
        print("\nfalse negatives:")
        for s, t in fn:
            print(f"  {s!r} -> {t!r}")
    if review:
        print(f"\nflagged for review ({len(review)}):")
        for s, t in sorted(review):
            print(f"  {s!r} -> {t!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
