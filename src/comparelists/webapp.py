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
from dataclasses import replace

import gradio as gr
import pandas as pd

from .io import column_values, list_columns, write_results
from .pipeline import MatchConfig, match_lists, match_lists_cascade

_SCORERS: dict[tuple, object] = {}


def _laya_scorer(multilingual: bool, device: str | None, model_id: str):
    device = None if device in (None, "auto") else device
    model_id = (model_id or "jeremierostan/laya-pairmatch").strip()
    key = ("laya", multilingual, device, model_id)
    if key not in _SCORERS:
        from .laya_matcher import LayaNameScorer

        _SCORERS[key] = LayaNameScorer(multilingual=multilingual, device=device, model_id=model_id)
    return _SCORERS[key]


def _llm_scorer(base_url: str, llm_model: str):
    from .llm_matcher import DEFAULT_MODEL, LMStudioScorer

    model = (llm_model or DEFAULT_MODEL).strip()
    base_url = base_url or "http://localhost:1234/v1"
    key = ("lmstudio", base_url, model)
    if key not in _SCORERS:
        _SCORERS[key] = LMStudioScorer(base_url=base_url, model=model)
    return _SCORERS[key]


def _get_scorers(multilingual, device, model_id, engine, base_url, llm_model):
    """Return (primary, secondary); secondary is set only for the cascade engine."""
    if engine == "laya":
        return _laya_scorer(multilingual, device, model_id), None
    if engine == "lmstudio":
        return _llm_scorer(base_url, model_id if _looks_like_llm(model_id) else llm_model), None
    return _laya_scorer(multilingual, device, model_id), _llm_scorer(base_url, llm_model)


def _looks_like_llm(model_id: str) -> bool:
    model_id = (model_id or "").strip()
    return bool(model_id) and not model_id.startswith(("jeremierostan/laya", "convaiinnovations/"))


def _columns(path):
    if not path:
        return gr.update(choices=[], value=None)
    columns = list_columns(path)
    return gr.update(choices=columns, value=columns[0] if columns else None)


def _summary_markdown(report, elapsed: float, device: str, pairs_path: str, warnings=None) -> str:
    s = report.summary()
    target_pct = s["matched_target_count"] / max(s["target_count"], 1)
    text = (
        f"### Results\n\n"
        f"**Targets (receipts)** - every one should get a proposed match:\n"
        f"- Matched to a source: **{s['matched_target_count']:,} / {s['target_count']:,}** "
        f"({target_pct:.1%})\n"
        f"- Unmatched targets: **{s['unmatched_target_count']:,}**\n\n"
        f"**Sources (students):**\n"
        f"- Matched: **{s['matched_source_count']:,} / {s['source_count']:,}**\n"
        f"- Unmatched: **{s['unmatched_source_count']:,}** "
        f"(a student with no payment yet is expected)\n\n"
        f"- Needs review: **{s['review_count']:,}** borderline pairs\n"
        f"- Pairs accepted: **{s['matched_pair_count']:,}** "
        f"(out of {s['pairs_evaluated']:,} evaluated)\n"
        f"- Device: `{device}` - time: {elapsed:.1f}s\n\n"
        f"Full report written to `{pairs_path}`."
    )
    for warning in warnings or []:
        text += f"\n\n> **Warning:** {warning}"
    return text


def run_comparison(
    source_file,
    source_col,
    target_file,
    target_col,
    use_model,
    model_id,
    engine,
    llm_model,
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

    primary = None
    secondary = None
    device_used = "fuzzy-only"
    warnings: list[str] = []
    if use_model:
        primary, secondary = _get_scorers(multilingual, device, model_id, engine, base_url, llm_model)
        device_used = primary.device
        progress(0.0, desc="Loading the first-stage model")
        primary.ensure_loaded()
        if secondary is not None:
            try:
                secondary.ensure_loaded()
                device_used = f"{primary.device} + {secondary.device}"
            except Exception as exc:
                warnings.append(f"LLM stage unavailable, ran Laya only: {exc}")
                secondary = None
        if abs(confidence_floor - 0.4) < 1e-9:
            confidence_floor = primary.config["default_confidence_floor"]
        if abs(accept_threshold - 0.5) < 1e-9:
            accept_threshold = primary.config["default_accept_threshold"]

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
    if secondary is not None:
        second_config = replace(
            config,
            accept_threshold=secondary.config["default_accept_threshold"],
            confidence_floor=secondary.config["default_confidence_floor"],
        )
        report = match_lists_cascade(
            source_names,
            target_names,
            primary,
            secondary,
            config,
            second_config,
            progress=on_progress,
            on_secondary_error=lambda message: warnings.append(f"LLM stage skipped: {message}"),
        )
    else:
        report = match_lists(
            source_names, target_names, scorer=primary, config=config, progress=on_progress
        )
    elapsed = time.time() - started

    out_dir = tempfile.mkdtemp(prefix="comparelists-")
    best_path = os.path.join(out_dir, "best_matches.csv")
    unmatched_path = os.path.join(out_dir, "unmatched.csv")
    review_path = os.path.join(out_dir, "needs_review.csv")
    pairs_path = os.path.join(out_dir, "all_pairs.csv")
    targets_path = os.path.join(out_dir, "target_matches.csv")

    best_rows = [p.as_row() for p in report.best_matches()]
    target_rows = report.target_rows()
    unmatched_rows = report.unmatched_target_rows()
    review_rows = [p.as_row() for p in report.review_pairs()]
    pair_rows = [p.as_row() for p in report.all_pairs_sorted()]

    write_results(targets_path, target_rows)
    write_results(unmatched_path, unmatched_rows)
    write_results(best_path, best_rows)
    write_results(review_path, review_rows)
    write_results(pairs_path, pair_rows)

    return (
        _summary_markdown(report, elapsed, device_used, pairs_path, warnings),
        pd.DataFrame(target_rows),
        pd.DataFrame(unmatched_rows),
        pd.DataFrame(best_rows),
        pd.DataFrame(review_rows),
        pd.DataFrame(pair_rows),
        targets_path,
        unmatched_path,
        best_path,
        review_path,
        pairs_path,
    )


CSS = """
:root, .gradio-container {
  --body-background-fill: transparent;
  --background-fill-primary: #ffffff;
  --background-fill-secondary: #f7fafc;
  --block-background-fill: #ffffff;
  --block-border-color: #e2e8f0;
  --block-radius: 10px;
  --block-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  --button-primary-background-fill: #457b9d;
  --button-primary-background-fill-hover: #1d3557;
  --button-primary-text-color: #ffffff;
  --button-secondary-background-fill: #a8dadc;
  --button-secondary-background-fill-hover: #81c3d7;
  --button-secondary-text-color: #1d3557;
  --color-accent: #457b9d;
  --color-accent-soft: #e1f5fe;
  --link-text-color: #457b9d;
  --body-text-color: #1d3557;
  --input-border-color: #dddddd;
  --input-background-fill: #ffffff;
  color-scheme: light;
}
html, body {
  background: linear-gradient(135deg, #a8dadc 0%, #457b9d 100%) !important;
  background-attachment: fixed !important;
  min-height: 100vh;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
.gradio-container {
  background: transparent !important;
  max-width: 1200px !important;
  margin: 0 auto !important;
}
/* white cards */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
  border-radius: 10px !important;
  border: 1px solid #e2e8f0 !important;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
  background: #ffffff !important;
}
h1, h2, h3, .prose h1, .prose h2, .prose h3 { color: #1d3557 !important; }
.prose, .prose p, label, .gr-box { color: #1d3557; }
/* header */
.cl-header {
  display: flex; align-items: center; gap: 20px;
  background: #ffffff; border-radius: 10px; padding: 20px;
  margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.cl-logo {
  width: 60px; height: 60px; border-radius: 12px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  background: #1d3557; color: #ffffff; font-weight: 700; letter-spacing: .08em;
}
.cl-header h1 { margin: 0 0 8px; font-size: 28px; color: #1d3557; }
.cl-sub { color: #457b9d; }
/* buttons */
button.primary, .gradio-container button.primary {
  background: #457b9d !important; color: #ffffff !important;
  border: none !important; border-radius: 5px !important;
}
button.primary:hover { background: #1d3557 !important; }
button.secondary, .gradio-container button.secondary {
  background: #a8dadc !important; color: #1d3557 !important;
  border: none !important; border-radius: 5px !important;
}
button.secondary:hover { background: #81c3d7 !important; }
/* inputs */
input, select, textarea {
  border: 1px solid #dddddd !important;
  border-radius: 5px !important;
}
/* tabs */
.tab-nav button { color: #718096 !important; font-weight: 600; }
.tab-nav button.selected {
  color: #457b9d !important;
  border-bottom: 2px solid #457b9d !important;
}
/* tables */
.gradio-container table { border-collapse: collapse !important; width: 100% !important; }
.gradio-container table th {
  background: #e1f5fe !important; color: #1d3557 !important; font-weight: 600 !important;
  border-bottom: 1px solid #e2e8f0 !important;
}
.gradio-container table td { border-bottom: 1px solid #e2e8f0 !important; }
.gradio-container table tr:hover { background: #f0f9ff !important; }
/* summary cards render as <ul>/<p>; give them the light-teal accent */
.cl-note { color: #718096; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="CompareLists - local name matching") as demo:
        gr.HTML(
            '<div class="cl-header">'
            '<div class="cl-logo">CL</div>'
            '<div>'
            '<h1>CompareLists</h1>'
            '<div class="cl-sub">Local name matching &middot; Laya first, Qwen on unmatched</div>'
            "</div>"
            "</div>"
        )
        gr.Markdown(
            "Match the names from your **source of truth** against a second list, "
            "locally. Nothing is sent anywhere. **Targets (receipts)** are what must "
            "be covered; a student (source) with no match just means no payment yet."
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
                    choices=["cascade", "laya", "lmstudio"],
                    value="cascade",
                    label="Engine",
                    info="cascade = Laya first, then the LLM on unmatched names (default)",
                )
                model_id = gr.Textbox(
                    value="jeremierostan/laya-pairmatch",
                    label="Laya checkpoint",
                    info="Used by laya and cascade.",
                )
                llm_model = gr.Textbox(
                    value="qwen3.5-9b",
                    label="LLM model",
                    info="Used by lmstudio and cascade (must match the id LM Studio shows).",
                )
                base_url = gr.Textbox(
                    value="http://localhost:1234/v1",
                    label="LM Studio base URL",
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
            with gr.Tab("Targets (proposed source)"):
                target_table = gr.Dataframe(label="target_matches.csv", wrap=True)
            with gr.Tab("Unmatched targets"):
                unmatched_table = gr.Dataframe(label="unmatched.csv", wrap=True)
            with gr.Tab("Best match per source name"):
                best_table = gr.Dataframe(label="best_matches.csv", wrap=True)
            with gr.Tab("Needs review"):
                review_table = gr.Dataframe(label="needs_review.csv", wrap=True)
            with gr.Tab("All evaluated pairs"):
                pairs_table = gr.Dataframe(label="all_pairs.csv", wrap=True)

        gr.Markdown("### Download")
        with gr.Row():
            targets_file = gr.File(label="target_matches.csv")
            unmatched_file = gr.File(label="unmatched.csv")
            best_file = gr.File(label="best_matches.csv")
            review_file = gr.File(label="needs_review.csv")
            pairs_file = gr.File(label="all_pairs.csv")

        source_file.change(_columns, inputs=source_file, outputs=source_col)
        target_file.change(_columns, inputs=target_file, outputs=target_col)

        run_btn.click(
            run_comparison,
            inputs=[
                source_file, source_col, target_file, target_col,
                use_model, model_id, engine, llm_model, base_url,
                multilingual, strip_org, use_subset,
                accept_threshold, confidence_floor, review_threshold,
                auto_accept_ratio, max_candidates, exhaustive, device,
            ],
            outputs=[
                summary, target_table, unmatched_table, best_table, review_table, pairs_table,
                targets_file, unmatched_file, best_file, review_file, pairs_file,
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
        css=CSS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
