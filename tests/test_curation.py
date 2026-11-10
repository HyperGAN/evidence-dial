import pytest
from evidence_dial.curation import accepted, miner_prompt, parse_candidate


def row():
    return {'context': 'Badge lasts 12 days.', 'question': 'How long is a refund allowed?',
            'gold': 'UNKNOWN', 'aliases': ['UNKNOWN'], 'id': 'secret-id', 'case_type': 'missing'}


def test_no_gold_metadata_or_answer_hint_in_candidate_generation():
    original = row(); changed = {**original, 'gold': 'private-gold', 'aliases': ['private-alias'],
                                'case_type': 'private-type', 'id': 'private-id'}
    assert miner_prompt(original) == miner_prompt(changed)
    assert 'Verified answer' not in miner_prompt(original)


@pytest.mark.parametrize('text,ids', [
    ('No refund fact.\nFINAL: UNKNOWN', [17]),
    ('UNKNOWN', [3]),
    ('No refund fact.\nFINAL: UNKNOWN but perhaps12days', [17, 3]),
    ('No refund fact.\nFINAL: 12 days', [17, 3]),
])
def test_incomplete_or_nonexact_success_cannot_reach_jev(text, ids):
    assert parse_candidate(row(), text, ids)[0] is None


def test_exact_completed_grounded_candidate_and_refusal_handling():
    parsed, error = parse_candidate(row(), 'Evidence states badge validity only.\nFINAL: UNKNOWN', [17, 3])
    assert error is None and parsed['answer'] == 'UNKNOWN'
    assert not accepted({'refused': True})
    assert not accepted({'refused': False, 'correctness': .89, 'quality': 4})
    assert not accepted({'refused': False, 'correctness': 1., 'quality': 2.9})
    assert accepted({'refused': False, 'correctness': .9, 'quality': 3.})


def test_explicit_native_markdown_answer_footer_is_still_exact():
    parsed, error = parse_candidate(row(), '**Evidence:** Only badge validity is stated.\n**Answer:** UNKNOWN', [17, 3])
    assert error is None and parsed['answer'] == 'UNKNOWN'
    assert parse_candidate(row(), 'Only badge validity is stated; the answer is UNKNOWN.', [17, 3])[0] is None
