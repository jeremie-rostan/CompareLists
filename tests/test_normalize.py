from comparelists.normalize import normalize, signature, tokens


def test_normalize_strips_accents_and_punctuation():
    assert normalize("Ana García-López") == "ana garcia lopez"
    assert normalize("  Jón   Þórsson ") == "jon thorsson"


def test_signature_is_order_insensitive():
    assert signature("Johnson, Katherine") == signature("Katherine Johnson")
    assert signature("Li Wei") == signature("Wei Li")


def test_tokens_drop_org_suffixes_when_asked():
    assert tokens("Acme Corporation Inc") == ["acme", "corporation", "inc"]
    assert tokens("Acme Corporation Inc", strip_org_suffixes=True) == ["acme"]


def test_normalize_handles_none_and_empty():
    assert normalize(None) == ""
    assert signature("") == ""
