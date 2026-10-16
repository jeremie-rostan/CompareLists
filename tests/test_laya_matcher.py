from comparelists.laya_matcher import (
    PRESETS,
    _base_state,
    _pairmatch_state,
    preset_for,
)
from comparelists.normalize import canonicalize_comment


def test_preset_is_autodetected_from_model_id():
    assert preset_for("convaiinnovations/laya") == "base"
    assert preset_for("jeremierostan/laya-pairmatch") == "pairmatch"
    assert preset_for("whatever", preset="base") == "base"


def test_presets_expose_required_keys():
    for preset in PRESETS.values():
        assert {"state", "questions", "answer_key", "prob_key"} <= set(preset)


def test_base_state_uses_source_target_keys():
    assert _base_state("Nikhil", "Raghunathan") == {
        "source_name": "Nikhil",
        "target_name": "Raghunathan",
    }


def test_pairmatch_state_uses_roster_payment_keys():
    assert _pairmatch_state("Raghunathan, Nikhil", "NIKHIL RAGHUNATHAN ($10)") == {
        "roster_name": "Raghunathan, Nikhil",
        "payment_comment": "NIKHIL RAGHUNATHAN ($10)",
    }


def test_canon_comment_matches_training_shape():
    assert canonicalize_comment("RAFAEL QUINTANA - Camila Quintana") == "RAFAEL QUINTANA -Camila Quintana ($0)"
    assert canonicalize_comment("A - B ($16)") == "A -B ($16)"
    assert canonicalize_comment("A- B") == "A -B ($0)"
    assert canonicalize_comment("PARENT -Child ($10)") == "PARENT -Child ($10)"


def test_canon_comment_keeps_hyphens_inside_names():
    assert canonicalize_comment("Raghunathan-Kovalenko - Arai Jun") == "Raghunathan-Kovalenko -Arai Jun ($0)"
