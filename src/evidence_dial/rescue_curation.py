"""Training-only semantic admission; frozen evaluation scoring is unchanged."""
import re

FOOTER = re.compile(r'\n\s*\*{0,2}(?:FINAL(?:\s+ANSWER)?|Answer)\s*:\*{0,2}\s*([^\n]+)\s*$', re.I)


def extract(saved):
    if not saved['output_ids'] or saved['output_ids'][-1] != 3:
        return None, 'not_native_EOS_complete'
    match = FOOTER.search(saved['text'])
    if not match or not saved['text'][:match.start()].strip():
        return None, 'no_worked_response_and_final_footer'
    answer = match[1].strip()
    if answer.startswith('**') or answer.endswith('**'):
        return None, 'answer_decoration_outside_qualified_native_suffix'
    return {'explanation': saved['text'][:match.start()].strip(), 'answer': answer,
            'answer_offset': match.start(1)}, None


def answer_start(saved, tok, parsed):
    """Keep literal sampled tokens/states, including terminators; no rewrite."""
    output = saved['output_ids']; offset = parsed['answer_offset']
    for i in range(len(output)):
        before = tok.decode(output[:i], skip_special_tokens=True)
        after = tok.decode(output[:i+1], skip_special_tokens=True)
        if len(before) <= offset < len(after):
            if before.rstrip() != saved['text'][:offset].rstrip():
                raise ValueError('A native token straddles the footer and answer')
            if tok.decode(output[i:], skip_special_tokens=True).strip() != parsed['answer']:
                raise ValueError('Actual sampled suffix differs from extracted answer')
            return i
    raise ValueError('Actual native answer boundary missing')


def fixtures():
    examples = [
        ('Standard film is exposed at 24 frames per second.', 'At what rate is film exposed?',
         '24 frames per second', 'The passage gives a rate of 24 frames per second.\nFINAL: 24 f/s',
         'The passage gives a rate of 24 frames per minute.\nFINAL: 24 f/min'),
        ('The Department created the Defense Clandestine Service (DCS) under the Defense Intelligence Agency.',
         'What service did the Department create?', 'the Defense Clandestine Service',
         'The passage names the newly created Defense Clandestine Service.\nFINAL: Defense Clandestine Service (DCS)',
         'The passage names the newly created Defense Intelligence Agency.\nFINAL: Defense Intelligence Agency'),
        ('Degrees Scheiner measure the sensitivity, or speed, of a photographic plate.',
         'What do degrees Scheiner indicate?', 'the speed of a plate',
         'The passage equates plate speed with sensitivity.\nFINAL: plate sensitivity',
         'The passage equates plate speed with physical plate size.\nFINAL: plate size'),
        ('The synthesis occurred in the 1930s and 1940s.', 'When did the synthesis occur?',
         'in the 1930s and 1940s', 'The passage dates the synthesis to both decades.\nFINAL: 1930s and 1940s',
         'The passage dates the synthesis to the 1920s and 1940s.\nFINAL: 1920s and 1940s'),
        ('Central Asian guerrillas, known as basmachi, fought the Bolshevik armies.',
         'Who fought the Bolshevik armies?', 'guerrillas throughout Central Asia, known as basmachi',
         'The passage names the guerrillas as basmachi.\nFINAL: basmachi',
         'The passage names the opponents of the Bolsheviks as the Bolsheviks themselves.\nFINAL: Bolsheviks'),
    ]
    return [{'row': {'context': c, 'question': q, 'gold': gold}, 'candidate': text, 'expected': expected}
            for c, q, gold, yes, no in examples for text, expected in [(yes, True), (no, False)]]
