import pytest
from evidence_dial.protocol import success, normalize, prompt


@pytest.mark.parametrize("gold,output,expected", [
    ("UNKNOWN", "UNKNOWN", 1), ("UNKNOWN", " Unknown.\n", 1),
    ("UNKNOWN", "Unknown, but probably 30 days", 0),
    ("UNKNOWN", "The answer is UNKNOWN", 0),
    ("UNKNOWN", "There is no information", 0),
    ("CONFLICT", "Conflict!", 1), ("CONFLICT", "No conflict", 0),
    ("CONFLICT", "CONFLICT: 7 or 9 days", 0),
    ("23 days", "23 DAYS.", 1), ("23 days", "23 days, or 30", 0),
    ("23 days", "123 days", 0), ("23 days", "23", 0),
    ("B", "B.", 1), ("B", "B because the evidence says so", 0),
    ("B", "The answer is B", 0),
])
def test_strict_scorer(gold, output, expected):
    assert success({"aliases": [gold]}, output) == expected


def test_mc_option_text_and_no_substring():
    row = {"aliases": ["D", "Planetary gravity will become stronger."]}
    assert success(row, "Planetary gravity will become stronger.") == 1
    assert success(row, "D. Planetary gravity will become stronger.") == 0


def test_missing_not_zero():
    with pytest.raises(TypeError):
        success({"aliases": ["UNKNOWN"]}, None)


def test_teacher_information_only_privileged():
    row = {"context": "A policy", "question": "How long?", "gold": "23 days"}
    assert "23 days" not in prompt(row)
    assert "Verified answer (private teacher information): 23 days" in prompt(row, teacher=True)
