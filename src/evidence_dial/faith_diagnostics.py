"""Diagnostic substring protocol, separate from the frozen strict selector.

FaithEval's published README supplies strict/non-strict QA substrings. Its
full multiple-choice evaluation script is not published there; MC diagnostics
below are explicitly option-text substrings, never a published leaderboard.
"""
from .protocol import faitheval_normalize, faith_diagnostics as qa_diagnostics


def faith_diagnostics(row, output):
    if row['case_type'] in {'missing','conflict'}:
        return {**qa_diagnostics(row,output), 'substring_protocol':'FaithEval published README QA normalization and phrase lists'}
    # A normalized option letter A is empty after article deletion; never let
    # an empty substring turn every generation into a diagnostic success.
    options=[faitheval_normalize(a) for a in row['aliases'] if len(a)>1]
    options=[a for a in options if a]
    match=int(any(a in faitheval_normalize(output) for a in options))
    return {'faitheval_strict_substring':match, 'faitheval_nonstrict_substring':match,
        'substring_protocol':'gold option text substring diagnostic; published full MC script unavailable; primary MC letter/text exact match remains separate'}
