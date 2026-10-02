"""Thin wrapper around local Laya decision models.

Two input "presets" are supported:

* ``base`` - the stock ``convaiinnovations/laya`` checkpoints, asked whether a
  target entry refers to a source person.
* ``pairmatch`` - the fine-tuned ``jeremierostan/laya-pairmatch`` checkpoint,
  which expects a ``roster_name`` / ``payment_comment`` state and a two-option
  ``same`` / ``different`` question.

The preset is chosen from the model id unless passed explicitly.

Only official ``laya`` / ``transformers`` APIs are used. The model is downloaded
once by ``huggingface_hub`` and then served entirely from disk, so no name ever
leaves the machine running the app.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

# transformers probes for TensorFlow at import; when TF is present its abseil
# runtime can deadlock model construction. Pin this before importing laya.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

BASE_QUESTIONS = {
    "same_entity": {
        "type": "choice",
        "instructions": (
            "The first string is a name from a source list. The second string is an "
            "entry from a target list and may contain extra words or other people (for "
            "example a parent next to a child, or siblings). Decide whether the target "
            "entry refers to the same person as the source name, allowing nicknames and "
            "short forms of the same first name (Bob = Robert, Bill = William, "
            "Liz = Elizabeth), different word order, spelling variations, accents, "
            "omitted middle names, initials and abbreviations."
        ),
        "criteria": {
            "same": (
                "the target refers to that same person, possibly as one of several "
                "people named, or with a slight spelling difference or nickname"
            ),
            "different": "the target refers to a different person",
            "uncertain": "not enough information to tell",
        },
    }
}

PAIRMATCH_QUESTIONS = {
    "match": {
        "type": "choice",
        "instructions": "Does this payment comment refer to the listed roster student?",
        "criteria": {
            "same": (
                "the comment refers to the same student, even with typos, truncation, "
                "nicknames, amounts, or family annotations"
            ),
            "different": (
                "the comment refers to a different person, a parent, a sibling, or "
                "someone else entirely"
            ),
        },
    },
}


def _base_state(name_a: str, name_b: str) -> dict:
    return {"source_name": name_a, "target_name": name_b}


def _pairmatch_state(name_a: str, name_b: str) -> dict:
    return {"roster_name": name_a, "payment_comment": name_b}


PRESETS: dict[str, dict] = {
    "base": {
        "state": _base_state,
        "questions": BASE_QUESTIONS,
        "answer_key": "same_entity",
        "prob_key": "same",
        "default_confidence_floor": 0.4,
    },
    "pairmatch": {
        "state": _pairmatch_state,
        "questions": PAIRMATCH_QUESTIONS,
        "answer_key": "match",
        "prob_key": "same",
        "default_confidence_floor": 0.0,
    },
}


def preset_for(model_id: str, preset: Optional[str] = None) -> str:
    if preset:
        return preset
    if "pairmatch" in model_id:
        return "pairmatch"
    return "base"


def detect_device(override: Optional[str] = None) -> str:
    """Pick the best available local accelerator."""
    if override:
        return override
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class LayaNameScorer:
    """Scores how likely two names refer to the same entity."""

    def __init__(
        self,
        multilingual: bool = False,
        device: Optional[str] = None,
        model_id: str = "convaiinnovations/laya",
        preset: Optional[str] = None,
    ):
        self.model_id = model_id
        self.preset = preset_for(model_id, preset)
        self.subfolder = "multilingual" if (multilingual and self.preset == "base") else None
        self.device = detect_device(device)
        self._agent = None
        self._lock = threading.Lock()
        self._load_seconds: Optional[float] = None

    @property
    def load_seconds(self) -> Optional[float]:
        return self._load_seconds

    @property
    def config(self) -> dict:
        return PRESETS[self.preset]

    def ensure_loaded(self):
        """Download (first run only) and construct the model if needed."""
        if self._agent is not None:
            return self._agent
        import laya

        started = time.time()
        kwargs = {"device": self.device}
        if self.subfolder:
            kwargs["subfolder"] = self.subfolder
        self._agent = laya.load(self.model_id, **kwargs)
        self._load_seconds = time.time() - started
        return self._agent

    def warmup(self):
        """Run one tiny prediction so the first real call is not slow."""
        self.score("A", "A")

    def score(self, name_a: str, name_b: str) -> dict:
        """Return calibrated probabilities for a pair of names."""
        agent = self.ensure_loaded()
        config = self.config
        state = config["state"](name_a, name_b)
        with self._lock:
            result = agent.predict(state, config["questions"], min_confidence=0.0)
        answer = result["answers"][config["answer_key"]]
        probs = answer.get("probabilities") or {}

        def prob(key: str) -> float:
            value = probs.get(key)
            return 0.0 if value is None else float(value)

        similarity = None
        if "similarity" in result.get("answers", {}):
            similarity = result["answers"]["similarity"].get("score")

        return {
            "same_prob": prob(config["prob_key"]),
            "choice": answer.get("choice"),
            "confidence": answer.get("confidence"),
            "similarity": similarity,
            "raw": answer,
        }
