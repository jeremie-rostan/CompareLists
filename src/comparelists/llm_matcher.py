"""Local LLM scorer backed by an OpenAI-compatible server (LM Studio).

This is an alternative "engine" to :mod:`comparelists.laya_matcher`. It asks a
local instruction model whether each candidate comment refers to a given roster
student, and returns a probability that plugs into the same pipeline.

Design notes:

* **One call per source.** The model receives the student plus all of that
  student's blocked candidates at once and returns a JSON verdict per candidate,
  which keeps a 7-9B model practical on a laptop.
* **Localhost only.** Requests go to ``http://localhost:1234/v1`` by default, so
  no name leaves the machine.
* No ``requests`` dependency: stdlib ``urllib`` is enough for local HTTP.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "qwen3.5-9b"
_JSON_INSTRUCTION = (
    'Return ONLY a JSON object of the form '
    '{"results":[{"index":0,"match":true,"confidence":0.0}]} '
    "with exactly one entry per candidate, in the same order."
)

SYSTEM_PROMPT = (
    "You are a precise record-matching assistant for a school. You are given one "
    "roster student and several payment-comment candidates. A payment comment is "
    "what a parent submitted; it usually contains the parent's own name and the "
    "name of one or more children, plus amounts and noise.\n\n"
    "A candidate refers to the student if it names them or clearly identifies them "
    "-- given name with one or more surnames, nicknames, spelling variants, "
    "initials, truncation, or a partial name. It does NOT refer to the student if "
    "it names a different child (for example a sibling with a different given "
    "name), a parent, or another family. Sharing only a surname, or only a common "
    "first name, is NOT enough.\n\n"
    "Be conservative: when the evidence is ambiguous, answer match=false with low "
    "confidence. Confidence is your probability (0-1) that the verdict is correct.\n\n"
    "Answer directly with the JSON object only - no reasoning, no explanation, no "
    "markdown, no code fences.\n\n"
    + _JSON_INSTRUCTION
)

# Few-shot examples: (student, [candidates], parsed_results)
FEW_SHOT: list[tuple[str, list[str], dict]] = [
    (
        "ALVAREZ MENDOZA, Mateo",
        ["CARLOS ALVAREZ -Mateo Alvarez Mendoza ($32)"],
        {"results": [{"index": 0, "match": True, "confidence": 0.98}]},
    ),
    (
        "GONZALEZ LOPEZ, Ana",
        ["CARLOS GONZALEZ -Sofia Gonzalez ($16)"],
        {"results": [{"index": 0, "match": False, "confidence": 0.96}]},
    ),
    (
        "LARRINAGA CARLES, Olivia",
        ["DAVID JOSUE LARRINAGA CHACIN -Olivia Larrinaga ($0)"],
        {"results": [{"index": 0, "match": True, "confidence": 0.9}]},
    ),
]


def _user_prompt(source: str, candidates: list[str]) -> str:
    lines = [f"Roster student: {source}", "Candidates:"]
    for idx, candidate in enumerate(candidates):
        lines.append(f"{idx}. {candidate}")
    lines.append("")
    lines.append(_JSON_INSTRUCTION)
    return "\n".join(lines)


def build_messages(source: str, candidates: list[str]) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for student, cands, results in FEW_SHOT:
        messages.append({"role": "user", "content": _user_prompt(student, cands)})
        messages.append({"role": "assistant", "content": json.dumps(results)})
    messages.append({"role": "user", "content": _user_prompt(source, candidates)})
    return messages


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in model output: {text!r}")
    return json.loads(text[start : end + 1])


def parse_results(text: str, count: int) -> list[dict]:
    """Turn raw model text into one verdict per candidate index."""
    data = _extract_json(text)
    raw = data.get("results") or []
    by_index: dict[int, dict] = {}
    for item in raw:
        try:
            by_index[int(item["index"])] = item
        except (KeyError, TypeError, ValueError):
            continue

    verdicts: list[dict] = []
    for idx in range(count):
        item = by_index.get(idx)
        if item is None:
            verdicts.append(
                {"same_prob": 0.0, "confidence": 0.0, "choice": "different", "reason": ""}
            )
            continue
        is_match = bool(item.get("match"))
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)
        verdicts.append(
            {
                "same_prob": confidence if is_match else 0.0,
                "confidence": confidence,
                "choice": "same" if is_match else "different",
                "reason": item.get("reason", ""),
            }
        )
    return verdicts


class LMStudioScorer:
    """Scores name/comment pairs with a local OpenAI-compatible LLM server."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 180.0,
        temperature: float = 0.0,
        api_key: str = "lm-studio",
        json_mode: bool = False,
        max_tokens: int = 512,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.api_key = api_key
        self.json_mode = json_mode
        self.max_tokens = max_tokens
        self.device = "lmstudio"
        self.preset = "llm"

    @property
    def config(self) -> dict:
        return {
            "default_accept_threshold": 0.5,
            "default_confidence_floor": 0.0,
        }

    def ensure_loaded(self):
        """Verify the local server is reachable and the model is available."""
        url = f"{self.base_url}/models"
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout, 15)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"Could not reach LM Studio at {self.base_url}. Start the local server "
                f"and load the model there. ({exc})"
            ) from exc
        ids = [m.get("id") for m in payload.get("data", [])]
        if ids and self.model not in ids:
            raise RuntimeError(
                f"Model {self.model!r} is not loaded in LM Studio. Available: {ids}"
            )
        return self

    def warmup(self):
        return None

    def score(self, name_a: str, name_b: str) -> dict:
        return self.score_many(name_a, [name_b])[0]

    def score_many(self, source: str, candidates: list[str]) -> list[dict]:
        if not candidates:
            return []
        body = {
            "model": self.model,
            "messages": build_messages(source, candidates),
            "temperature": self.temperature,
            "stream": False,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.max_tokens and self.max_tokens > 0:
            body["max_tokens"] = self.max_tokens
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"LM Studio returned {exc.code} for model {self.model!r}: {detail[:300]}"
            ) from exc
        content = payload["choices"][0]["message"]["content"]
        return parse_results(content, len(candidates))
