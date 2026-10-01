# CompareLists

Match a **source-of-truth list of names** against a **second list**, entirely on
your own machine, using the local [Laya](https://huggingface.co/convaiinnovations/laya)
decision model.

- Handles different word order (`Johnson, Katherine` = `Katherine Johnson`),
  accents (`García` = `Garcia`), spelling variants, nicknames (`Bob` = `Robert`)
  and abbreviations (`Acme Corp` = `Acme Corporation`).
- Laya is a non-generative decision model: it returns calibrated probabilities,
  never invents text, and runs on-device after a one-time download.
- Ship a web UI (drag-and-drop CSVs) **and** a command-line tool.
- Nothing is uploaded. There is no API, no account, no server.

Output tells you which source names were found, which second-list name they
matched, the model's probability and confidence, plus three review files:
`best_matches.csv`, `needs_review.csv`, `unmatched.csv`, `all_pairs.csv`.

---

## How it works

1. **Normalise** every name: lowercase, de-accent, transliterate (`Þ` → `th`),
   strip punctuation. An order-insensitive *signature* catches exact matches
   without touching the model.
2. **Block**: for each source name, RapidFuzz keeps only the top-K most similar
   target names, so we never run the model on the full Cartesian product.
3. **Decide**: each surviving pair is sent to Laya as a three-way `choice`
   question — *same entity* / *different entity* / *uncertain*. A pair is
   accepted only when Laya answers **same**, the probability clears
   `accept_threshold`, and the answer clears `confidence_floor`. Pairs that are
   plausible but not confident enough land in `needs_review.csv`.

The model is downloaded once to the Hugging Face cache
(`~/.cache/huggingface`) and reused afterwards. Everything after that is offline.

---

## Requirements

- macOS on Apple Silicon (an NVIDIA GPU or plain CPU also work; auto-detected)
- [`uv`](https://docs.astral.sh/uv/) (one-line install: `brew install uv`)
- ~2 GB free disk for the model, plus the Python environment

## Setup (once)

```bash
git clone <your-repo-url> CompareLists
cd CompareLists
uv sync --extra dev
```

The first prediction downloads the model (~30-60 s). Every run after that starts
in a few seconds.

## Use it — command line

```bash
# 1. Find the exact column names in your CSVs
uv run comparelists source.csv target.csv --list-columns

# 2. Match
uv run comparelists source.csv target.csv \
  --source-col full_name --target-col attendee \
  --output-dir comparison-results
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--accept-threshold` | Laya `same` probability needed to accept (default `0.5`) |
| `--confidence-floor` | minimum Laya confidence to accept (default `0.25`) |
| `--review-threshold` | flag borderline pairs above this (default `0.4`) |
| `--auto-accept-ratio` | fuzzy score that skips the model (default `97`) |
| `--max-candidates` | targets considered per source name (default `10`) |
| `--exhaustive` | score every pair, for small lists (slower, most accurate) |
| `--strip-org-suffixes` | ignore `Ltd`, `Inc`, `Corporation`… |
| `--no-model` | fuzzy-only, no model download |
| `--multilingual` | use the multilingual checkpoint for non-Latin scripts |
| `--device` | `mps`, `cuda` or `cpu` (default: auto-detect) |

The example lists live in `examples/` — try:

```bash
uv run comparelists examples/source.csv examples/target.csv \
  --source-col full_name --target-col attendee
```

## Use it — web interface

```bash
uv run comparelists-web              # http://127.0.0.1:7860 on this Mac only
uv run comparelists-web --host 0.0.0.0   # reachable by other devices on your network
uv run comparelists-web --share      # temporary public Gradio link (compute stays here)
```

In the browser: upload both CSVs, pick the name column for each, then click
**Find matches**. Results and the four CSVs are downloadable.

---

## Sharing it with other people

Because the model must run locally, the cleanest way to share is **GitHub**:
people clone the repo and run it on their own machine, so their data also never
leaves their computer.

```bash
# from this folder, once
git add .
git commit -m "CompareLists: local name matching with Laya"
gh repo create comparelists --public --source=. --push   # needs the GitHub CLI
# (or: create an empty repo on github.com, then)
# git remote add origin git@github.com:<you>/comparelists.git
# git push -u origin main
```

Then anyone runs:

```bash
git clone https://github.com/<you>/comparelists.git
cd comparelists
uv sync
uv run comparelists-web
```

If you'd rather not publish code but still let people use the tool while compute
stays on your Mac, use `uv run comparelists-web --host 0.0.0.0` (same Wi-Fi) or
`--share` (public link tunneled to your running process). Do **not** deploy the
model to a third-party host if the data must stay local.

---

## Tuning matching quality

- **False positives?** Raise `--accept-threshold` and/or `--confidence-floor`.
- **Missing semantic matches** (nicknames, very different spellings)? Raise
  `--max-candidates` and use `--exhaustive` on small lists; lower
  `--auto-accept-ratio` so more pairs reach the model.
- **Non-Latin scripts?** Add `--multilingual`.
- Always check `needs_review.csv` for borderline cases.

Real examples from `examples/`:

| Source | Second list | Result |
| --- | --- | --- |
| `Katherine Johnson` | `Johnson, Katherine` | exact |
| `Mohammed Al-Rashid` | `Muhammad Al Rashid` | Laya 0.95 |
| `Robert Smith` | `Bob Smith` | Laya 0.86 |
| `Acme Corporation` | `Acme Corp` | Laya 0.86 |
| `Jón Þórsson` | `Jon Thorsson` | exact (transliteration) |

## Privacy

- No network calls after the one-time model download.
- No telemetry from this app; no names written outside your output directory.
- Set `HF_HUB_OFFLINE=1` to prove it runs offline once the model is cached.

## Development

```bash
uv run pytest              # unit tests (use a fake scorer, no model download)
uv run python -m comparelists --help
```

Project layout:

```
src/comparelists/
  normalize.py      de-accent, transliterate, order-insensitive signatures
  pipeline.py       blocking + decision logic + report dataclasses
  laya_matcher.py   local Laya wrapper (model load, choice question, probabilities)
  io.py             CSV read/write
  cli.py            `comparelists` command
  webapp.py         `comparelists-web` Gradio UI
```

## Notes and limits

- The shipped checkpoint is used zero-shot. On its own benchmark Laya is a fast
  base to specialise rather than a turnkey decision engine; here it is combined
  with deterministic exact/transliteration handling and RapidFuzz blocking.
- Laya warns that some shipped temperatures are invalid, so `confidence` can be
  treated as a relative signal rather than a perfectly calibrated value.
- Scoring cost is roughly 30-60 ms per pair on Apple Silicon. Candidate blocking
  keeps a 1,000 × 1,000 comparison in the low thousands of model calls.
