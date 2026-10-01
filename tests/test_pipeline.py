"""Pipeline tests use a fake scorer so they never download the model."""

from comparelists.pipeline import MatchConfig, match_lists


class FakeScorer:
    """Returns a scripted probability per (normalised) pair."""

    def __init__(self, probabilities=None, default=0.0):
        self.probabilities = probabilities or {}
        self.default = default
        self.calls = 0

    def ensure_loaded(self):
        return self

    def warmup(self):
        pass

    def score(self, a, b):
        self.calls += 1
        key = frozenset((a.lower(), b.lower()))
        same = self.probabilities.get(key, self.default)
        return {"same_prob": same, "choice": "same" if same >= 0.5 else "different", "confidence": same}


def test_exact_order_insensitive_match_skips_model():
    scorer = FakeScorer()
    report = match_lists(
        ["Katherine Johnson"],
        ["Johnson, Katherine"],
        scorer=scorer,
        config=MatchConfig(),
    )
    best = report.best_matches()
    assert len(best) == 1
    assert best[0].method == "exact"
    assert best[0].decision
    assert scorer.calls == 0


def test_model_decides_on_gray_zone_pair():
    scorer = FakeScorer({frozenset(("bob smith", "robert smith")): 0.91})
    report = match_lists(
        ["Robert Smith"],
        ["Bob Smith"],
        scorer=scorer,
        config=MatchConfig(accept_threshold=0.6, auto_accept_ratio=99.0),
    )
    best = report.best_matches()[0]
    assert best.method == "laya"
    assert best.same_prob == 0.91
    assert best.decision


def test_below_threshold_is_unmatched():
    scorer = FakeScorer({frozenset(("jon smith", "jane doe")): 0.1})
    report = match_lists(
        ["Jon Smith"],
        ["Jane Doe"],
        scorer=scorer,
        config=MatchConfig(accept_threshold=0.6, auto_accept_ratio=99.0),
    )
    assert report.best_matches() == []
    assert report.unmatched == [(0, "Jon Smith")]


def test_high_fuzzy_auto_accepts_without_model():
    scorer = FakeScorer()
    report = match_lists(
        ["Jonathon"],
        ["Jonathan"],
        scorer=scorer,
        config=MatchConfig(auto_accept_ratio=85.0),
    )
    assert report.best_matches()[0].decision
    assert report.best_matches()[0].method == "fuzzy-high"
    assert scorer.calls == 0


def test_best_match_per_source_returns_highest_combined():
    scorer = FakeScorer(
        {
            frozenset(("robert smith", "bob smith")): 0.8,
            frozenset(("robert smith", "rob smith")): 0.95,
        }
    )
    report = match_lists(
        ["Robert Smith"],
        ["Bob Smith", "Rob Smith"],
        scorer=scorer,
        config=MatchConfig(accept_threshold=0.5, auto_accept_ratio=99.0, max_candidates=5),
    )
    best = report.best_matches()[0]
    assert best.target == "Rob Smith"


def test_outputs_are_alphabetical():
    scorer = FakeScorer()
    report = match_lists(
        ["Zoe", "Ana", "Mike"],
        ["Zoe", "Ana"],
        scorer=scorer,
        config=MatchConfig(),
    )
    assert [p.source for p in report.best_matches()] == ["Ana", "Zoe"]
    assert [name for _, name in report.unmatched_sorted()] == ["Mike"]
    assert [p.source for p in report.all_pairs_sorted()] == sorted(
        p.source for p in report.pairs
    )


def test_summary_counts():
    scorer = FakeScorer({frozenset(("robert smith", "bob smith")): 0.9})
    report = match_lists(
        ["Robert Smith", "Nobody Here"],
        ["Bob Smith", "Someone Else"],
        scorer=scorer,
        config=MatchConfig(accept_threshold=0.6, auto_accept_ratio=99.0),
    )
    summary = report.summary()
    assert summary["source_count"] == 2
    assert summary["matched_source_count"] == 1
    assert summary["unmatched_source_count"] == 1
