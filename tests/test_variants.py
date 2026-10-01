from comparelists.normalize import normalize
from comparelists.variants import expand_variants


def _normed(variants):
    return {normalize(v) for v in variants}


def test_spans_include_every_contiguous_run_and_the_full_cell():
    variants = expand_variants("María González Pedro Ramírez", mode="spans", max_tokens=4)
    got = _normed(variants)
    assert "pedro ramirez" in got
    assert "maria gonzalez" in got
    assert "maria gonzalez pedro ramirez" in got
    assert "gonzalez pedro" in got


def test_conjunctions_split_on_y_and_and():
    assert _normed(expand_variants("Juan Pérez y Lucía Pérez", mode="conjunctions")) >= {
        "juan perez",
        "lucia perez",
    }
    assert _normed(expand_variants("Kathy and Erica Thompson", mode="conjunctions")) >= {
        "kathy",
        "erica thompson",
    }


def test_spans_also_survive_conjunctions():
    got = _normed(expand_variants("Kathy and Erica", mode="spans", max_tokens=3))
    assert "kathy" in got
    assert "erica" in got
    assert "kathy and erica" in got


def test_none_mode_is_untouched():
    assert expand_variants("Pedro Ramírez", mode="none") == ["Pedro Ramírez"]


def test_single_token_cell_returns_itself():
    assert expand_variants("Pedro", mode="spans") == ["Pedro"]


def test_variants_are_deduped():
    variants = expand_variants("Ana Ana", mode="spans", max_tokens=2)
    assert len(variants) == len(_normed(variants))
