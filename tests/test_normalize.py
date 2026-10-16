from comparelists.normalize import normalize, signature, tokens


def test_normalize_strips_accents_and_punctuation():
    assert normalize("Camila Ibarra-Bonnet") == "camila ibarra bonnet"
    assert normalize("  Þórdís   Eiríksdóttir ") == "thordis eiriksdottir"


def test_signature_is_order_insensitive():
    assert signature("Raghunathan, Nikhil") == signature("Nikhil Raghunathan")
    assert signature("Arai Jun") == signature("Jun Arai")


def test_tokens_drop_org_suffixes_when_asked():
    assert tokens("Vespera Corporation Inc") == ["vespera", "corporation", "inc"]
    assert tokens("Vespera Corporation Inc", strip_org_suffixes=True) == ["vespera"]


def test_normalize_handles_none_and_empty():
    assert normalize(None) == ""
    assert signature("") == ""
