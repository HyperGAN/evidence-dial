"""Per-example train-side scaffolds; ordinary inference questions stay fixed."""
from .protocol import digest, messages

STRATEGIES = {
    'evidence-span': (
        'First locate the evidence span that directly answers the requested attribute. '
        'Quote its decisive phrase briefly. Check the subject, scope and units against the question.'),
    'candidate-search': (
        'Search the supplied records for candidate answers. List plausible candidates as hypotheses, '
        'with their supporting phrases. Eliminate wrong-scope or unsupported candidates; '
        'consider evidence for each remaining candidate before deciding.'),
    'scope-table': (
        'Break the question into subject, requested attribute, scope and units. '
        'Make a compact record-by-record comparison of those fields. '
        'Separate irrelevant records, agreeing records and incompatible values for the same scope.'),
    'challenge-answer': (
        'Form a tentative answer from the evidence, explicitly labelling it tentative. '
        'Then search every supplied record for a contradiction, missing support or scope mismatch. '
        'Reject a tentative answer if that check fails, and state the decisive evidence for your final choice.'),
}
PROTOCOL = 'per-example-train-side-evidence-reasoning-strategies-v1'


def strategy_for(case_id, trial):
    """Four strategies x four trials, rotated by ID; never use labels or gold."""
    if not 0 <= trial < 16:
        raise ValueError('Only the declared 16-attempt budget is available')
    names = tuple(STRATEGIES)
    offset = int(digest({'case_id': case_id, 'schedule_seed': 20261111})[:8], 16) % len(names)
    return names[(offset + trial) % len(names)]


def strategy_spec(name):
    return {'protocol': PROTOCOL, 'id': name, 'instructions': STRATEGIES[name]}


def training_prompt(row, name):
    return (
        'Use only the supplied evidence, overriding everyday knowledge. All records have equal authority. '
        'An absent requested fact requires UNKNOWN; incompatible values for the same scope require CONFLICT. '
        'Do not infer facts merely from their plausibility.\n\n'
        'Training reasoning procedure:\n' + STRATEGIES[name] + '\n\n'
        'Write compact working notes (at most 90 words), then a separate final line: '
        'FINAL: followed by the shortest supported answer with units, UNKNOWN or CONFLICT. '
        'Do not put further text after the final answer.\n\n'
        f"Evidence:\n{row['context']}\n\nQuestion: {row['question']}\n")


def validate_saved_strategy(row, selection, saved):
    name = selection['strategy_id']
    spec = strategy_spec(name)
    assert name == strategy_for(row['id'], saved['trial'])
    assert saved['id'] == row['id']
    assert selection['seed'] == saved['identity']['seed']
    assert selection['strategy_sha256'] == digest(spec)
    assert saved['identity']['strategy'] == spec
    assert saved['identity']['protocol'] == PROTOCOL
    assert saved['identity']['prompt'] == training_prompt(row, name)


def positive_messages(row, selection):
    """Context control retains each example's chosen train-side procedure."""
    text = messages(row)[0]['content']
    assert text.endswith('Answer:')
    return [{'role': 'user', 'content': text[:-len('Answer:')] +
             'Training reasoning procedure:\n' + STRATEGIES[selection['strategy_id']] + '\n' +
             'Verified successful model working (private training context):\n' +
             selection['candidate'] + '\nAnswer:'}]
