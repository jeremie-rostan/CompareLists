import io
import json
from unittest import mock

from comparelists.llm_matcher import (
    LMStudioScorer,
    build_messages,
    parse_results,
)


def test_build_messages_includes_source_and_candidates():
    messages = build_messages("Larrinaga Carles, Olivia", ["PARENT -Olivia Larrinaga ($0)"])
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "Larrinaga Carles, Olivia" in messages[-1]["content"]
    assert "0. PARENT -Olivia Larrinaga ($0)" in messages[-1]["content"]


def test_parse_results_maps_match_and_confidence():
    text = json.dumps(
        {
            "results": [
                {"index": 0, "match": True, "confidence": 0.9},
                {"index": 1, "match": False, "confidence": 0.8},
            ]
        }
    )
    verdicts = parse_results(text, 2)
    assert verdicts[0]["same_prob"] == 0.9 and verdicts[0]["choice"] == "same"
    assert verdicts[1]["same_prob"] == 0.0 and verdicts[1]["choice"] == "different"


def test_parse_results_tolerates_code_fences_and_missing_entries():
    text = '```json\n{"results": [{"index": 0, "match": true, "confidence": 0.7}]}\n```'
    verdicts = parse_results(text, 3)
    assert verdicts[0]["same_prob"] == 0.7
    assert verdicts[1]["same_prob"] == 0.0
    assert verdicts[2]["same_prob"] == 0.0


def test_parse_results_clamps_confidence():
    text = '{"results": [{"index": 0, "match": true, "confidence": 4.2}]}'
    assert parse_results(text, 1)[0]["same_prob"] == 1.0


def test_score_many_posts_and_parses():
    payload = {
        "choices": [
            {"message": {"content": '{"results":[{"index":0,"match":true,"confidence":0.85}]}'}}
        ]
    }
    fake = io.BytesIO(json.dumps(payload).encode("utf-8"))

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return fake.read()

    scorer = LMStudioScorer(base_url="http://localhost:1234/v1", model="test-model")
    with mock.patch("urllib.request.urlopen", return_value=_Resp()) as patched:
        results = scorer.score_many("Garcia, Ana", ["PARENT -Ana Garcia ($16)"])
    assert results[0]["same_prob"] == 0.85
    assert patched.called
