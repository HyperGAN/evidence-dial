from evidence_dial.faith_diagnostics import faith_diagnostics
from evidence_dial.protocol import success


def test_a_option_does_not_become_empty_substring_success():
    row={'case_type':'altered','aliases':['A','Gravity becomes stronger']}
    assert faith_diagnostics(row,'B')['faitheval_strict_substring']==0
    assert faith_diagnostics(row,'irrelevant prose')['faitheval_strict_substring']==0
    assert faith_diagnostics(row,'Gravity becomes stronger')['faitheval_strict_substring']==1
    assert success(row,'A')==1


def test_published_qa_substring_is_diagnostic_not_strict_selector():
    row={'case_type':'missing','aliases':['UNKNOWN']}
    assert faith_diagnostics(row,'UNKNOWN, but perhaps 30 days')['faitheval_strict_substring']==1
    assert success(row,'UNKNOWN, but perhaps 30 days')==0
