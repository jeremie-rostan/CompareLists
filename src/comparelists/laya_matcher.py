"""Thin wrapper around the local Laya decision model.

Only official ``laya`` / ``transformers`` APIs are used. The model is downloaded
once by ``huggingface_hub`` and then served entirely from disk, so no name ever
leaves the machine running the app.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

# transformers probes for TensorFlow at import; when TF is present its abseil
# runtime can deadlock model construction. Pin this before importing laya.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

QUESTION = {
    "same_entity": {
        "type": "choice",
        "instructions": (
            "The first string is a name from a source list. The second string is an "
            "entry from a target list and may contain one or more names, a parent's "
            "name next to a child's, extra words, or a slight variation in spelling. "
            "Does the target entry refer to the same person as the source name?"
        ),
        "criteria": {
            "same": (
                "the target refers to that same person, even if it is only one of "
                "several names listed or a slight spelling variation"
            ),
            "different": "the target does not refer to that person",
            "uncertain": "not enough information to tell",
        },
    }
}


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
    ):
        self.model_id = model_id
        self.subfolder = "multilingual" if multilingual else None
        self.device = detect_device(device)
        self._agent = None
        self._lock = threading.Lock()
        self._load_seconds: Optional[float] = None

    @property
    def load_seconds(self) -> Optional[float]:
        return self._load_seconds

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
        state = {"source_name": name_a, "target_name": name_b}
        with self._lock:
            result = agent.predict(state, QUESTION, min_confidence=0.0)
        answer = result["answers"]["same_entity"]
        probs = answer.get("probabilities") or {}

        def prob(key: str) -> float:
            value = probs.get(key)
            return 0.0 if value is None else float(value)

        return {
            "same_prob": prob("same"),
            "different_prob": prob("different"),
            "uncertain_prob": prob("uncertain"),
            "choice": answer.get("choice"),
            "confidence": answer.get("confidence"),
            "raw": answer,
        }
