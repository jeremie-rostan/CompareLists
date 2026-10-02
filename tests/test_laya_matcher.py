from comparelists.laya_matcher import (
    PRESETS,
    _base_state,
    _canon_comment,
    _pairmatch_state,
    preset_for,
)


def test_preset_is_autodetected_from_model_id():
    assert preset_for("convaiinnovations/laya") == "base"
    assert preset_for("jeremierostan/laya-pairmatch") == "pairmatch"
    assert preset_for("whatever", preset="base") == "base"


def test_presets_expose_required_keys():
    for preset in PRESETS.values():
        assert {"state", "questions", "answer_key", "prob_key"} <= set(preset)


def test_base_state_uses_source_target_keys():
    assert _base_state("Ana", "García") == {"source_name": "Ana", "target_name": "García"}


def test_pairmatch_state_uses_roster_payment_keys():
    assert _pairmatch_state("Nguyen, Linh", "LINH NGUYEN ($10)") == {
        "roster_name": "Nguyen, Linh",
        "payment_comment": "LINH NGUYEN ($10)",
    }


def test_canon_comment_matches_training_shape():
    assert _canon_comment("CELINA REVOREDO - Alegre Revoredo") == "CELINA REVOREDO -Alegre Revoredo ($0)"
    assert _canon_comment("A - B ($16)") == "A -B ($16)"
    assert _canon_comment("A- B") == "A -B ($0)"
    assert _canon_comment("PARENT -Child ($10)") == "PARENT -Child ($10)"


def test_canon_comment_keeps_hyphens_inside_names():
    assert _canon_comment("Wei-Lun - Chen Mei") == "Wei-Lun -Chen Mei ($0)"
