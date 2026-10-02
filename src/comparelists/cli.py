"""Command-line entry point: ``comparelists``."""

from __future__ import annotations

import os
import sys

# Offline support must be decided before huggingface_hub is imported anywhere,
# because it freezes these flags at import time.
if "--offline" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import time

from . import __version__
from .io import column_values, list_columns, write_results
from .pipeline import MatchConfig, match_lists


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comparelists",
        description=(
            "Find names from a source-of-truth list inside a second list, "
            "locally, using the Laya decision model."
        ),
    )
    parser.add_argument("source", nargs="?", help="CSV with the source-of-truth names")
    parser.add_argument("target", nargs="?", help="CSV with the list to search")
    parser.add_argument("--source-col", help="column holding names in the source CSV")
    parser.add_argument("--target-col", help="column holding names in the target CSV")
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="print each file's columns and exit (use this first to find your column names)",
    )
    parser.add_argument("--output-dir", default="comparison-results", help="where to write CSVs")
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--candidate-floor", type=float, default=0.0)
    parser.add_argument(
        "--accept-threshold",
        type=float,
        default=None,
        help="model probability to accept a match (default: per-checkpoint value)",
    )
    parser.add_argument(
        "--confidence-floor",
        type=float,
        default=None,
        help="minimum model confidence to accept (default: per-checkpoint value)",
    )
    parser.add_argument("--review-threshold", type=float, default=0.4)
    parser.add_argument("--auto-accept-ratio", type=float, default=97.0)
    parser.add_argument("--model-weight", type=float, default=0.5)
    parser.add_argument("--exhaustive", action="store_true", help="score every pair (slow)")
    parser.add_argument("--no-model", action="store_true", help="fuzzy-only, skip Laya")
    parser.add_argument(
        "--no-subset",
        action="store_true",
        help="disable deterministic token-containment matching (reversed/extra-name cases)",
    )
    parser.add_argument("--multilingual", action="store_true", help="use the multilingual checkpoint")
    parser.add_argument("--device", default=None, help="mps, cuda or cpu (auto-detected)")
    parser.add_argument(
        "--model",
        default="jeremierostan/laya-pairmatch",
        help="Laya checkpoint id or local path (default: the fine-tuned student matcher)",
    )
    parser.add_argument(
        "--preset",
        choices=["auto", "base", "pairmatch"],
        default="auto",
        help="input schema for the checkpoint; auto-detected from --model by default",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="download the model, then exit (run once online to enable --offline later)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never touch the network; fail if the model is not already cached",
    )
    parser.add_argument("--strip-org-suffixes", action="store_true")
    parser.add_argument("--delimiter", default=None, help="input delimiter (auto-detected)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _prefetch(args) -> int:
    from .laya_matcher import LayaNameScorer

    scorer = LayaNameScorer(
        multilingual=args.multilingual,
        device=args.device,
        model_id=args.model,
        preset=None if args.preset == "auto" else args.preset,
    )
    print(f"downloading Laya ({scorer.model_id}) on {scorer.device}...")
    started = time.time()
    scorer.ensure_loaded()
    print(f"model cached in {time.time() - started:.1f}s")
    print("now run with --offline to work without a network connection")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_columns:
        for path in (args.source, args.target):
            print(f"{path}:")
            for column in list_columns(path, args.delimiter):
                print(f"  - {column}")
        return 0

    if args.prefetch:
        return _prefetch(args)

    if not args.source or not args.target:
        print("error: provide a source CSV and a target CSV (try --help)", file=sys.stderr)
        return 2

    if not args.source_col or not args.target_col:
        print("error: --source-col and --target-col are required (try --list-columns)", file=sys.stderr)
        return 2

    source_names = column_values(args.source, args.source_col, args.delimiter)
    target_names = column_values(args.target, args.target_col, args.delimiter)

    print(f"source: {len(source_names):,} names from {args.source!r} [{args.source_col}]")
    print(f"target: {len(target_names):,} names from {args.target!r} [{args.target_col}]")

    scorer = None
    if not args.no_model:
        from .laya_matcher import LayaNameScorer

        scorer = LayaNameScorer(
            multilingual=args.multilingual,
            device=args.device,
            model_id=args.model,
            preset=None if args.preset == "auto" else args.preset,
        )
        print(f"loading Laya on {scorer.device} (first run downloads the model)...")
        started = time.time()
        scorer.ensure_loaded()
        print(f"model ready in {time.time() - started:.1f}s")

    confidence_floor = args.confidence_floor
    if confidence_floor is None:
        confidence_floor = scorer.config["default_confidence_floor"] if scorer else 0.0

    accept_threshold = args.accept_threshold
    if accept_threshold is None:
        accept_threshold = scorer.config["default_accept_threshold"] if scorer else 0.5

    config = MatchConfig(
        max_candidates=args.max_candidates,
        candidate_floor=args.candidate_floor,
        accept_threshold=accept_threshold,
        confidence_floor=confidence_floor,
        review_threshold=args.review_threshold,
        auto_accept_ratio=args.auto_accept_ratio,
        use_model=not args.no_model,
        exhaustive=args.exhaustive,
        strip_org_suffixes=args.strip_org_suffixes,
        model_weight=args.model_weight,
        subset_match=not args.no_subset,
    )

    started = time.time()
    report = match_lists(source_names, target_names, scorer=scorer, config=config)

    os.makedirs(args.output_dir, exist_ok=True)
    pairs_path = os.path.join(args.output_dir, "all_pairs.csv")
    best_path = os.path.join(args.output_dir, "best_matches.csv")
    review_path = os.path.join(args.output_dir, "needs_review.csv")
    unmatched_path = os.path.join(args.output_dir, "unmatched.csv")

    write_results(pairs_path, [p.as_row() for p in report.all_pairs_sorted()])
    write_results(best_path, [p.as_row() for p in report.best_matches()])
    write_results(review_path, [p.as_row() for p in report.review_pairs()])
    write_results(
        unmatched_path,
        [{"source_name": name, "source_row": row} for row, name in report.unmatched_sorted()],
    )

    summary = report.summary()
    print("\n--- summary ---")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"elapsed: {time.time() - started:.1f}s")
    print("\nwritten:")
    print(f"  {pairs_path}       (every pair evaluated)")
    print(f"  {best_path}        (best match per source name)")
    print(f"  {review_path}      (borderline pairs worth a human look)")
    print(f"  {unmatched_path}   (source names with no match)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
