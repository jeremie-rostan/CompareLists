"""Gradio web UI for CompareLists.

Run locally:

    uv run comparelists-web            # http://127.0.0.1:7860
    uv run comparelists-web --host 0.0.0.0   # reachable by other devices on your network

All inference stays on the machine that launches this process.
"""

from __future__ import annotations

import os
import sys

# Offline support must be decided before gradio / huggingface_hub are imported,
# because they freeze these flags at import time.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
if "--offline" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import tempfile

import gradio as gr
import pandas as pd

from .io import column_values, list_columns, write_results
from .pipeline import MatchConfig, match_lists

_SCORERS: dict[tuple, object] = {}


def _get_scorer(multilingual: bool, device: str | None, model_id: str, engine: str, base_url: str):
    device = None if device in (None, "auto") else device
    model_id = (model_id or "").strip()
    if engine == "lmstudio":
        from .llm_matcher import DEFAULT_MODEL, LMStudioScorer

        if not model_id or model_id.startswith(("jeremierostan/laya", "convaiinnovations/")):
            model_id = DEFAULT_MODEL
        key = ("lmstudio", base_url, model_id)
        if key not in _SCORERS:
            _SCORERS[key] = LMStudioScorer(base_url=base_url or "http://localhost:1234/v1", model=model_id)
        return _SCORERS[key]

    model_id = model_id or "jeremierostan/laya-pairmatch"
    key = ("laya", multilingual, device, model_id)
    if key not in _SCORERS:
        from .laya_matcher import LayaNameScorer

        _SCORERS[key] = LayaNameScorer(multilingual=multilingual, device=device, model_id=model_id)
    return _SCORERS[key]


def _columns(path):
    if not path:
        return gr.update(choices=[], value=None)
    columns = list_columns(path)
    return gr.update(choices=columns, value=columns[0] if columns else None)


def _summary_markdown(report, elapsed: float, device: str, pairs_path: str) -> str:
    s = report.summary()
    return (
        f"### Results\n\n"
        f"- Source names: **{s['source_count']:,}**\n"
        f"- Matched: **{s['matched_source_count']:,}** "
        f"({s['matched_source_count'] / max(s['source_count'], 1):.1%})\n"
        f"- Unmatched: **{s['unmatched_source_count']:,}**\n"
        f"- Needs review: **{s['review_count']:,}** borderline pairs\n"
        f"- Source/target pairs accepted: **{s['matched_pair_count']:,}** "
        f"(out of {s['pairs_evaluated']:,} evaluated)\n"
        f"- Device: `{device}` - time: {elapsed:.1f}s\n\n"
        f"Full report written to `{pairs_path}`."
    )


def run_comparison(
    source_file,
    source_col,
    target_file,
    target_col,
    use_model,
    model_id,
    engine,
    base_url,
    multilingual,
    strip_org,
    use_subset,
    accept_threshold,
    confidence_floor,
    review_threshold,
    auto_accept_ratio,
    max_candidates,
    exhaustive,
    device,
    progress=gr.Progress(),
):
    if not source_file or not target_file:
        raise gr.Error("Upload both a source CSV and a target CSV.")
    if not source_col or not target_col:
        raise gr.Error("Pick the name column for each file.")

    source_names = column_values(source_file, source_col)
    target_names = column_values(target_file, target_col)
    if not source_names or not target_names:
        raise gr.Error("One of the selected columns is empty.")

    scorer = None
    device_used = "fuzzy-only"
    if use_model:
        scorer = _get_scorer(multilingual, device, model_id, engine, base_url)
        device_used = scorer.device
        progress(0.0, desc="Loading Laya (first run downloads the model)")
        scorer.ensure_loaded()
        if abs(confidence_floor - 0.4) < 1e-9:
            confidence_floor = scorer.config["default_confidence_floor"]
        if abs(accept_threshold - 0.5) < 1e-9:
            accept_threshold = scorer.config["default_accept_threshold"]

    config = MatchConfig(
        max_candidates=int(max_candidates),
        accept_threshold=accept_threshold,
        confidence_floor=confidence_floor,
        review_threshold=review_threshold,
        auto_accept_ratio=auto_accept_ratio,
        use_model=use_model,
        exhaustive=exhaustive,
        strip_org_suffixes=strip_org,
        subset_match=use_subset,
    )

    def on_progress(current, total, name):
        progress(current / max(total, 1), desc=f"{current}/{total}: {name}")

    import time

    started = time.time()
    report = match_lists(source_names, target_names, scorer=scorer, config=config, progress=on_progress)
    elapsed = time.time() - started

    out_dir = tempfile.mkdtemp(prefix="comparelists-")
    best_path = os.path.join(out_dir, "best_matches.csv")
    unmatched_path = os.path.join(out_dir, "unmatched.csv")
    review_path = os.path.join(out_dir, "needs_review.csv")
    pairs_path = os.path.join(out_dir, "all_pairs.csv")

    best_rows = [p.as_row() for p in report.best_matches()]
    unmatched_rows = report.unmatched_with_candidates()
    review_rows = [p.as_row() for p in report.review_pairs()]
    pair_rows = [p.as_row() for p in report.all_pairs_sorted()]

    write_results(best_path, best_rows)
    write_results(unmatched_path, unmatched_rows)
    write_results(review_path, review_rows)
    write_results(pairs_path, pair_rows)

    return (
        _summary_markdown(report, elapsed, device_used, pairs_path),
        pd.DataFrame(best_rows),
        pd.DataFrame(unmatched_rows),
        pd.DataFrame(review_rows),
        pd.DataFrame(pair_rows),
        best_path,
        unmatched_path,
        review_path,
        pairs_path,
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="CompareLists - local name matching") as demo:
        gr.Markdown(
            "# CompareLists\n"
            "Find the names from your **source of truth** inside a second list, "
            "locally. Matching handles different order, spelling and accents using "
            "the [Laya](https://huggingface.co/convaiinnovations/laya) decision model "
            "running on this machine. No name is sent anywhere."
        )

        with gr.Row():
            with gr.Column():
                source_file = gr.File(label="Source of truth (CSV)", file_types=[".csv", ".tsv"])
                source_col = gr.Dropdown(label="Name column (source)", choices=[], interactive=True)
            with gr.Column():
                target_file = gr.File(label="Second list (CSV)", file_types=[".csv", ".tsv"])
                target_col = gr.Dropdown(label="Name column (second list)", choices=[], interactive=True)

        with gr.Accordion("Matching options", open=False):
            with gr.Row():
                engine = gr.Dropdown(
                    choices=["laya", "lmstudio"],
                    value="laya",
                    label="Engine",
                    info="laya (fast decision model) or lmstudio (local LLM server)",
                )
                model_id = gr.Textbox(
                    value="jeremierostan/laya-pairmatch",
                    label="Model",
                    info="Laya checkpoint id/path, or LLM model id for the lmstudio engine.",
                )
                base_url = gr.Textbox(
                    value="http://localhost:1234/v1",
                    label="LM Studio base URL (lmstudio engine)",
                )
            with gr.Row():
                use_model = gr.Checkbox(value=True, label="Use a model (uncheck for fast fuzzy-only)")
                multilingual = gr.Checkbox(value=False, label="Multilingual checkpoint")
                strip_org = gr.Checkbox(value=False, label="Ignore company suffixes (Ltd, Inc...)")
                use_subset = gr.Checkbox(
                    value=True,
                    label="Deterministic token matching (reversed names, extra/parent names)",
                )
                exhaustive = gr.Checkbox(value=False, label="Score every pair (slow, small lists only)")
            with gr.Row():
                accept_threshold = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.01,
                    label="Accept threshold (Laya same-entity probability)",
                )
                confidence_floor = gr.Slider(
                    0.0, 1.0, value=0.4, step=0.01,
                    label="Confidence floor (minimum Laya confidence to accept)",
                )
                review_threshold = gr.Slider(
                    0.0, 1.0, value=0.4, step=0.01,
                    label="Review threshold (flag borderline pairs above this)",
                )
            with gr.Row():
                auto_accept_ratio = gr.Slider(
                    0.0, 100.0, value=97.0, step=0.5,
                    label="Auto-accept fuzzy score (skip model above this)",
                )
                max_candidates = gr.Slider(1, 50, value=10, step=1, label="Candidates per source name")
                device = gr.Dropdown(
                    choices=["auto", "mps", "cuda", "cpu"], value="auto", label="Compute device"
                )

        run_btn = gr.Button("Find matches", variant="primary")

        summary = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("Best match per source name"):
                best_table = gr.Dataframe(label="best_matches.csv", wrap=True)
            with gr.Tab("Needs review"):
                review_table = gr.Dataframe(label="needs_review.csv", wrap=True)
            with gr.Tab("Unmatched source names"):
                unmatched_table = gr.Dataframe(label="unmatched.csv", wrap=True)
            with gr.Tab("All evaluated pairs"):
                pairs_table = gr.Dataframe(label="all_pairs.csv", wrap=True)

        gr.Markdown("### Download")
        with gr.Row():
            best_file = gr.File(label="best_matches.csv")
            review_file = gr.File(label="needs_review.csv")
            unmatched_file = gr.File(label="unmatched.csv")
            pairs_file = gr.File(label="all_pairs.csv")

        source_file.change(_columns, inputs=source_file, outputs=source_col)
        target_file.change(_columns, inputs=target_file, outputs=target_col)

        run_btn.click(
            run_comparison,
            inputs=[
                source_file, source_col, target_file, target_col,
                use_model, model_id, engine, base_url, multilingual, strip_org, use_subset,
                accept_threshold, confidence_floor, review_threshold,
                auto_accept_ratio, max_candidates, exhaustive, device,
            ],
            outputs=[
                summary, best_table, unmatched_table, review_table, pairs_table,
                best_file, unmatched_file, review_file, pairs_file,
            ],
        )
    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="comparelists-web", description="Launch the CompareLists web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to share on your LAN)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never touch the network; requires the model to be cached first",
    )
    args = parser.parse_args(argv)

    build_ui().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
