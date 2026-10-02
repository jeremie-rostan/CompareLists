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
`best_matches.csv`, `needs_review.csv`, `unmatched.csv`, `all_pairs.csv`. All of
them are sorted alphabetically by source name. `unmatched.csv` also lists each
unmatched name's **closest candidate** with its scores, so it doubles as a review
queue.

---

## How it works

1. **Normalise** every name: lowercase, de-accent, transliterate (`Þ` → `th`),
   strip punctuation. An order-insensitive *signature* catches exact matches
   without touching the model.
2. **Deterministic pass**: if every token of the source name appears in a target
   entry (or vice versa), it is a match. This reliably resolves reversed names,
   extra parent/sibling names, extra or omitted surnames and middle names without
   the model — and needs at least two tokens on each side, so a lone common first
   name cannot match.
3. **Block**: for each remaining source name, RapidFuzz keeps only the top-K most
   similar target entries, so we never run the model on the full Cartesian
   product.
4. **Decide**: each surviving pair is sent to Laya as a three-way `choice`
   question — does this target entry refer to this source person? (same /
   different / uncertain). The entry may hold a parent next to a child, siblings,
   extra words, a nickname or a spelling variation. A pair is accepted only when
   Laya answers **same**, the probability clears `accept_threshold`, and the
   answer clears `confidence_floor`. Plausible-but-uncertain pairs land in
   `needs_review.csv` instead of being reported as matches.

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
| `--no-subset` | disable the deterministic token-containment pass |
| `--multilingual` | use the multilingual checkpoint for non-Latin scripts |
| `--model` | Laya checkpoint id or local path (default `jeremierostan/laya-pairmatch`) |
| `--preset` | input schema: `auto` (default), `base` or `pairmatch` |
| `--device` | `mps`, `cuda` or `cpu` (default: auto-detect) |
| `--prefetch` | download the model, then exit (no CSV needed) |
| `--offline` | never touch the network; fail if the model is not cached |

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

### Working offline

The first run needs a network connection to download the model (English
checkpoint, ~1.6 GB) into the Hugging Face cache. After that, everything works
with wifi off.

To download deliberately, then run offline:

```bash
uv run comparelists --prefetch     # one online run: fetch and cache the model
uv run comparelists examples/source.csv examples/target.csv \
  --source-col full_name --target-col attendee --offline
uv run comparelists-web --offline  # same for the web UI
```

`--offline` sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` before anything is
imported, so the process fails loudly instead of silently reaching the network.
The web UI also disables Gradio analytics.

## Choosing a model

The default is
[`jeremierostan/laya-pairmatch`](https://huggingface.co/jeremierostan/laya-pairmatch),
a fine-tune of `convaiinnovations/laya-multilingual` trained to match parent
payment comments to roster students (siblings, shared surnames, nicknames,
typos, truncation, amounts, annotations, and surname-only references). Its own
held-out test set scores **0.9493 accuracy** (n=296), and it reaches **~0.93
end-to-end** through this app's pipeline on that distribution.

Any Laya-compatible checkpoint can be used with `--model` (CLI) or the **Model
checkpoint** field (web UI). The input schema is auto-detected, or set explicitly
with `--preset base|pairmatch`.

```bash
uv run comparelists source.csv target.csv \
  --source-col full_name --target-col comment --offline
# use the stock checkpoint instead:
uv run comparelists source.csv target.csv \
  --source-col full_name --target-col attendee \
  --model convaiinnovations/laya --offline
```

On the deliberately out-of-distribution labelled set in `evaluation/` (bare name
pairs, no parent comment), the checkpoints trade off as follows:

| Checkpoint | recall | precision | misses | false positive |
| --- | --- | --- | --- | --- |
| `jeremierostan/laya-pairmatch` (default) | 0.95 | 0.86 | — | `Smith` vs `Roberta Smith` |
| `convaiinnovations/laya` (base) | 1.00 | 0.95 | none | shared-surname trap |

The base checkpoint has marginally better recall on *bare* name pairs; pairmatch
is far stronger on the real payment-comment distribution. **Test on your real
data** — `evaluation/eval_matching.py [model_id]` measures whichever is
configured.

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

## Messy target columns (parent + student, siblings)

Rosters often put more than one name in a single target cell:

```
"María González Pedro Ramírez"     parent first, then the student
"Juan Pérez y Lucía Pérez"         two siblings joined by "y"
"Kathy and Erica Thompson"         two students joined by "and"
"Roberto Sánchez, Ana Sánchez"     comma separated
```

You do **not** need to split these columns. The model is asked, for each source
name, whether a target entry *refers to that person* - the entry may hold a
parent next to one or more children, several names, extra words, or a slight
spelling variation. It reads the whole entry and answers same / different /
uncertain, so a student is found even when the cell starts with a parent.

Try it:

```bash
uv run comparelists examples/parents_source.csv examples/parents_target.csv \
  --source-col full_name --target-col household
```

| Source | Target entry | Result |
| --- | --- | --- |
| `Pedro Ramírez` | `María González Pedro Ramírez` | match |
| `Lucía Pérez` | `Juan Pérez y Lucía Pérez` | match |
| `Erica Thompson` | `Kathy and Erica Thompson` | match |
| `Ana Sánchez` | `Roberto Sánchez, Ana Sánchez` | match |

The `matched_entry` column shows the original target cell so you can audit each
match.

## Tuning matching quality

- **False positives?** Raise `--accept-threshold` and/or `--confidence-floor`.
  The defaults (`0.5` / `0.4`) accept only confident model matches; anything
  plausible but uncertain goes to `needs_review.csv` rather than being reported
  as a match.
- **Missing semantic matches** (nicknames, unusual spellings that the
  deterministic pass cannot catch)? Raise `--max-candidates` and use
  `--exhaustive` on small lists.
- **Non-Latin scripts?** Add `--multilingual`.
- Always check `needs_review.csv` for borderline cases, and `all_pairs.csv` to
  audit any decision.

## Reliability

There is a labelled evaluation set and a harness you can run after any change to
the model, wording or thresholds:

```bash
uv run python evaluation/eval_matching.py [model_id]
```

On the out-of-distribution labelled set (bare name pairs, no parent comment),
the default `pairmatch` checkpoint scores **recall 0.950, precision 0.864** at
its own tuned threshold. That set is intentionally harsher than real data: it
has no parent comment, so nickname and transliteration cases are hard. On the
model's own payment-comment distribution it scores **~0.93** end-to-end, including
comments that name the student by surnames only. Each checkpoint has its own
default acceptance threshold (`pairmatch` 0.4, `base` 0.5).

No automatic system is perfect. Always sanity-check `needs_review.csv`, and audit
`all_pairs.csv` for decisions that share a surname but differ in the given name —
that is the hardest class for any of the checkpoints.

## Privacy

- No network calls after the one-time model download.
- No telemetry from this app; no names written outside your output directory.
- Set `HF_HUB_OFFLINE=1` (or pass `--offline`) to prove it runs offline once the
  model is cached.

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
evaluation/
  labeled_pairs.csv labelled name pairs with expected verdicts
  eval_matching.py  precision/recall harness (real model)
```

## Notes and limits

- The shipped checkpoint is used zero-shot. On its own benchmark Laya is a fast
  base to specialise rather than a turnkey decision engine; here it is combined
  with deterministic exact/transliteration handling and RapidFuzz blocking.
- Laya warns that some shipped temperatures are invalid, so `confidence` can be
  treated as a relative signal rather than a perfectly calibrated value.
- Scoring cost is roughly 30-60 ms per pair on Apple Silicon. Candidate blocking
  keeps a 1,000 × 1,000 comparison in the low thousands of model calls.
