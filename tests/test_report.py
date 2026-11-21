import pytest
from evidence_dial.report import paired, paired_abstention


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


def test_grounding_abstention_uses_strict_matched_absence_and_answerable_probes():
    rows = [
        {'id': 'missing', 'case_type': 'missing', 'aliases': ['UNKNOWN']},
        {'id': 'answerable', 'case_type': 'answerable', 'aliases': ['30 days']},
        {'id': 'unmeasured', 'case_type': 'missing', 'aliases': ['UNKNOWN']},
    ]
    off = [dict(pred('missing', 1), output=' Unknown. '),
           dict(pred('answerable', 1), output='30 days'),
           dict(pred('unmeasured', 1), output='UNKNOWN')]
    on = [dict(pred('missing', 0), output='UNKNOWN, but probably 30 days'),
          dict(pred('answerable', 0), output='UNKNOWN')]
    audit = paired_abstention(rows, on, off)
    assert audit['missing_evidence_strict_unknown'] == {
        'declared': 2, 'measured_pairs': 1, 'complete': False,
        'on_rate': 0.0, 'off_rate': 1.0, 'on_minus_off': -1.0,
    }
    assert audit['answerable_evidence_abstention']['on_minus_off'] == 1.0
    assert audit['used_for_selection'] is False
    assert paired_abstention(rows, off, off)['missing_evidence_strict_unknown']['on_minus_off'] == 0


def test_grounding_abstention_rejects_mismatched_prompts_and_incorrect_saved_scores():
    rows = [{'id': 'missing', 'case_type': 'missing', 'aliases': ['UNKNOWN']}]
    off = [dict(pred('missing', 1), output='UNKNOWN')]
    with pytest.raises(ValueError, match='prompt mismatch'):
        paired_abstention(rows, [dict(off[0], prompt_sha256='different')], off)
    with pytest.raises(ValueError, match='strict score differs'):
        paired_abstention(rows, [dict(off[0], output='UNKNOWN? likely 30 days')], off)
