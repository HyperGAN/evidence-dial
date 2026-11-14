"""Extra training-only search contexts, with answer assistance labelled."""
import re


def prompt(row, trial):
    if row['split'] != 'train' or not 16 <= trial < 48:
        raise ValueError('Only additional trials16..47 of declared training cases')
    strategies = [
        'Match the question subject literally before looking at values. Copy only facts about that exact subject and requested attribute. Ignore other subjects and unrelated inventory counters. Multiple different values for the SAME subject and attribute mean CONFLICT, even inside one source. One unique supported value is the answer.',
        'Find the smallest passage span answering the question. Read neighbouring sentences to resolve references. Quote the decisive span, explain it in one short sentence, and copy the answer with units. Equivalent names or abbreviations are allowed; do not add unrelated claims.',
        'Build a tiny list of (subject, attribute, value) from every relevant statement. Filter by the requested subject AND attribute, then compare the remaining values. Do not confuse absence, disagreement and unrelated distractors. Keep the explanation under45 words.',
    ]
    text = ('Use only the evidence. All records have equal authority. An absent fact means UNKNOWN; '
            'incompatible answers for the same scope mean CONFLICT.\n\nTraining search procedure:\n' + strategies[trial % 3])
    entity = re.search(r'\bunit\s+([A-Za-z]\d+)\b', row['question'], re.I)
    if entity:
        text += '\nThe exact unit named in the question is: ' + entity[1] + '. Other unit identifiers are different subjects.'
    clarification = row['id'] == 'squad-train-5726ea77f1498d1400e8efc5'
    if clarification:
        text += '\nTraining-only clarification of the incomplete wording: identify what groups use Utrecht\'s waterways.'
    assisted = trial >= 40
    if assisted:
        text += ('\nTraining-only reference answer to verify: ' + row['gold'] +
                 '\nFind actual support for this candidate in the passage. Do not repeat it if the evidence does not justify it.')
    text += ('\n\nGive a short grounded explanation, then a separate line FINAL: followed by the answer. '
             'End after that line.\n\nEvidence:\n' + row['context'] + '\n\nQuestion: ' + row['question'] + '\n')
    return text, {'protocol': 'additional-training-only-mining-rescue-v1', 'procedure': trial % 3,
                  'reference_answer_supplied': assisted, 'question_clarification_supplied': clarification}
