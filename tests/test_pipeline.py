"""Pipeline tests use a fake scorer so they never download the model."""

from comparelists.pipeline import MatchConfig, match_lists, match_lists_cascade


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
        ["Nikhil Raghunathan"],
        ["Raghunathan, Nikhil"],
        scorer=scorer,
        config=MatchConfig(),
    )
    best = report.best_matches()
    assert len(best) == 1
    assert best[0].method == "exact"
    assert best[0].decision
    assert scorer.calls == 0


def test_model_decides_on_gray_zone_pair():
    scorer = FakeScorer({frozenset(("toby reinhardt", "tobias reinhardt")): 0.91})
    report = match_lists(
        ["Tobias Reinhardt"],
        ["Toby Reinhardt"],
        scorer=scorer,
        config=MatchConfig(accept_threshold=0.6, auto_accept_ratio=99.0),
    )
    best = report.best_matches()[0]
    assert best.method == "laya"
    assert best.same_prob == 0.91
    assert best.decision


def test_below_threshold_is_unmatched():
    scorer = FakeScorer({frozenset(("callum fraser", "fiona fraser")): 0.1})
    report = match_lists(
        ["Callum Fraser"],
        ["Fiona Fraser"],
        scorer=scorer,
        config=MatchConfig(accept_threshold=0.6, auto_accept_ratio=99.0),
    )
    assert report.best_matches() == []
    assert report.unmatched == [(0, "Callum Fraser")]


def test_high_fuzzy_auto_accepts_without_model():
    scorer = FakeScorer()
    report = match_lists(
        ["Kovalenko"],
        ["Kovalenk"],
        scorer=scorer,
        config=MatchConfig(auto_accept_ratio=85.0),
    )
    assert report.best_matches()[0].decision
    assert report.best_matches()[0].method == "fuzzy-high"
    assert scorer.calls == 0


def test_best_match_per_source_returns_highest_combined():
    scorer = FakeScorer(
        {
            frozenset(("tobias reinhardt", "toby reinhardt")): 0.8,
            frozenset(("tobias reinhardt", "tobi reinhardt")): 0.95,
        }
    )
    report = match_lists(
        ["Tobias Reinhardt"],
        ["Toby Reinhardt", "Tobi Reinhardt"],
        scorer=scorer,
        config=MatchConfig(accept_threshold=0.5, auto_accept_ratio=99.0, max_candidates=5),
    )
    best = report.best_matches()[0]
    assert best.target == "Tobi Reinhardt"


def test_outputs_are_alphabetical():
    scorer = FakeScorer()
    report = match_lists(
        ["Zara", "Amara", "Miles"],
        ["Zara", "Amara"],
        scorer=scorer,
        config=MatchConfig(),
    )
    assert [p.source for p in report.best_matches()] == ["Amara", "Zara"]
    assert [name for _, name in report.unmatched_sorted()] == ["Miles"]
    assert [p.source for p in report.all_pairs_sorted()] == sorted(
        p.source for p in report.pairs
    )


def test_subset_matches_reversed_and_noisy_target_without_model():
    scorer = FakeScorer()
    report = match_lists(
        ["Camila Ibarra"],
        ["Ibarra Osvaldo, Camila Ibarra"],
        scorer=scorer,
        config=MatchConfig(),
    )
    best = report.best_matches()[0]
    assert best.method == "subset"
    assert best.decision
    assert scorer.calls == 0


def test_subset_tolerates_extra_surnames_and_omitted_middle_names():
    scorer = FakeScorer()
    for source, target in [
        ("Camila Ibarra", "Camila Ibarra Fuentes"),
        ("Mireya Lucía Moreno", "Moreno, Mireya Lucía"),
        ("Mateo Quintana", "Rafael Quintana Mateo Quintana"),
    ]:
        report = match_lists([source], [target], scorer=scorer, config=MatchConfig())
        assert report.best_matches()[0].decision, (source, target)
    assert scorer.calls == 0


def test_single_token_source_is_not_auto_matched_by_subset():
    scorer = FakeScorer()
    report = match_lists(["Sasha"], ["Sasha Novak"], scorer=scorer, config=MatchConfig())
    assert all(p.method != "subset" for p in report.pairs)


def test_unmatched_includes_closest_candidate():
    scorer = FakeScorer({frozenset(("tobias reinhardt", "toby reinhardt")): 0.1})
    report = match_lists(
        ["Tobias Reinhardt"],
        ["Toby Reinhardt"],
        scorer=scorer,
        config=MatchConfig(accept_threshold=0.6, auto_accept_ratio=99.0),
    )
    rows = report.unmatched_with_candidates()
    assert rows[0]["source_name"] == "Tobias Reinhardt"
    assert rows[0]["closest_candidate"] == "Toby Reinhardt"
    assert rows[0]["closest_laya_same_prob"] == 0.1


def test_unmatched_with_empty_target_has_no_candidate():
    report = match_lists(["Gerald"], [], config=MatchConfig(use_model=False))
    rows = report.unmatched_with_candidates()
    assert rows == [{"source_name": "Gerald", "source_row": 0}]


class BatchScorer:
    """Scores all candidates for a source in one call (like the LLM engine)."""

    def __init__(self):
        self.batch_calls = 0

    def ensure_loaded(self):
        return self

    def warmup(self):
        pass

    def score(self, a, b):  # must not be used
        raise AssertionError("per-pair score() should not be called")

    def score_many(self, source, candidates):
        self.batch_calls += 1
        out = []
        for candidate in candidates:
            hit = source == "Tobias Reinhardt" and candidate == "Toby Reinhardt"
            out.append(
                {
                    "same_prob": 0.95 if hit else 0.02,
                    "confidence": 0.95 if hit else 0.02,
                    "choice": "same" if hit else "different",
                }
            )
        return out


def test_batched_scorer_is_used_once_per_source():
    scorer = BatchScorer()
    report = match_lists(
        ["Tobias Reinhardt"],
        ["Toby Reinhardt", "Fiona Fraser"],
        scorer=scorer,
        config=MatchConfig(accept_threshold=0.5, auto_accept_ratio=99.0),
    )
    best = report.best_matches()[0]
    assert best.target == "Toby Reinhardt"
    assert best.method == "llm"
    assert scorer.batch_calls == 1


def test_cascade_runs_secondary_only_on_unmatched_names():
    primary = FakeScorer(default=0.0)
    secondary = BatchScorer()
    report = match_lists_cascade(
        ["Tobias Reinhardt", "Nobody Here"],
        ["Toby Reinhardt", "Fiona Fraser"],
        primary,
        secondary,
        MatchConfig(accept_threshold=0.6, auto_accept_ratio=99.0),
        MatchConfig(accept_threshold=0.5, auto_accept_ratio=99.0),
    )
    matches = {p.source: p.target for p in report.best_matches()}
    assert matches["Tobias Reinhardt"] == "Toby Reinhardt"
    assert report.unmatched == [(1, "Nobody Here")]


def test_cascade_skips_secondary_when_primary_matches_everything():
    primary = FakeScorer({frozenset(("tobias reinhardt", "toby reinhardt")): 0.9})
    secondary = BatchScorer()
    report = match_lists_cascade(
        ["Tobias Reinhardt"],
        ["Toby Reinhardt"],
        primary,
        secondary,
        MatchConfig(accept_threshold=0.6, auto_accept_ratio=99.0),
        MatchConfig(),
    )
    assert report.best_matches()[0].target == "Toby Reinhardt"
    assert secondary.batch_calls == 0


def test_target_centric_unmatched_and_proposal():
    scorer = FakeScorer({frozenset(("tobias reinhardt", "toby reinhardt")): 0.9})
    report = match_lists(
        ["Tobias Reinhardt"],
        ["Toby Reinhardt", "Fiona Fraser"],
        scorer=scorer,
        config=MatchConfig(accept_threshold=0.6, auto_accept_ratio=99.0),
    )
    by_entry = {row["target_entry"]: row for row in report.target_rows()}
    assert by_entry["Toby Reinhardt"]["used"] is True
    assert by_entry["Toby Reinhardt"]["proposed_source"] == "Tobias Reinhardt"
    assert by_entry["Fiona Fraser"]["used"] is False

    unmatched = report.unmatched_target_rows()
    assert [row["target_entry"] for row in unmatched] == ["Fiona Fraser"]
    # the proposed candidate for an unmatched target comes from the SOURCE list
    assert unmatched[0]["proposed_source"] == "Tobias Reinhardt"

    summary = report.summary()
    assert summary["target_count"] == 2
    assert summary["matched_target_count"] == 1
    assert summary["unmatched_target_count"] == 1


def test_summary_counts():
    scorer = FakeScorer({frozenset(("tobias reinhardt", "toby reinhardt")): 0.9})
    report = match_lists(
        ["Tobias Reinhardt", "Nobody Here"],
        ["Toby Reinhardt", "Someone Else"],
        scorer=scorer,
        config=MatchConfig(accept_threshold=0.6, auto_accept_ratio=99.0),
    )
    summary = report.summary()
    assert summary["source_count"] == 2
    assert summary["matched_source_count"] == 1
    assert summary["unmatched_source_count"] == 1
