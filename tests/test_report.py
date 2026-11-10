import pytest
from evidence_dial.report import paired


def pred(id, score, prompt="same"):
    return {"id": id, "success": score, "prompt_sha256": prompt, "teacher": False}


def test_effect_is_equal_case_delta_and_coverage_is_required():
    rows = [{"id": "a", "family": "one"}, {"id": "b", "family": "one"}, {"id": "c", "family": "two"}]
    off = [pred("a", 0), pred("b", 1), pred("c", 0)]
    on = [pred("a", 1), pred("b", 0), pred("c", 1)]
    r = paired(rows, on, off, bootstrap=100)
    assert r["complete"] and r["effect_gain"] == pytest.approx(1/3)
    assert r["gains"] == 2 and r["losses"] == 1
    r = paired(rows, on[:-1], off, bootstrap=100)
    assert not r["complete"] and r["missing_on"] == ["c"]


def test_mismatched_prompts_cannot_make_a_win():
    with pytest.raises(ValueError, match="prompt mismatch"):
        paired([{"id":"a", "family":"f"}], [pred("a",1,"strong")], [pred("a",0,"weak")])
