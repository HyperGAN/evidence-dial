"""Train-only real QA plus controlled evidence contrasts, frozen before calls."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from .io import read_jsonl, write_json, write_jsonl
from .protocol import digest, prompt

SEED = 20261111


def words(text):
    return re.findall(r'\w+', text.casefold())


def shingles(text, width=18):
    tokens = words(text)
    return {hashlib.sha256(' '.join(tokens[i:i+width]).encode()).digest()
            for i in range(len(tokens)-width+1)}


def oracle(facts, scope, field):
    values = {f['value'] for f in facts if f['scope'] == scope and f['field'] == field}
    return 'UNKNOWN' if not values else 'CONFLICT' if len(values) > 1 else next(iter(values))


def real_candidates(source, prior):
    old_ids = {str(r.get('source_id', r['id'])) for r in prior}
    old_questions = {' '.join(words(r['question'])) for r in prior}
    old_shingles = set().union(*(shingles(r['context']) for r in prior))
    allowed, excluded = [], []
    for article in source['data']:
        overlap = None
        candidates = []
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            if shingles(context) & old_shingles:
                overlap = 'shared_18_word_context_span'
            for qa in paragraph['qas']:
                if qa['id'] in old_ids or ' '.join(words(qa['question'])) in old_questions:
                    overlap = 'prior_question_or_source_id'
                answers = list(dict.fromkeys(a['text'] for a in qa['answers']))
                if (not 55 <= len(words(context)) <= 160 or not answers
                        or any(not a or len(a) > 56 or len(a.split()) > 8 for a in answers)):
                    continue
                assert all(context[a['answer_start']:a['answer_start']+len(a['text'])] == a['text'] for a in qa['answers'])
                candidates.append({'source_id': qa['id'], 'context': context,
                                   'question': qa['question'], 'gold': answers[0], 'aliases': answers,
                                   'article': article['title']})
        # Exclude the entire article if ANY paragraph/question overlaps a prior
        # research test. Never consult model outputs or success rates here.
        if overlap:
            excluded.append({'article': article['title'], 'reason': overlap})
        elif candidates:
            allowed.append((article['title'], candidates))
    return allowed, excluded


def synthetic(split, kind, index):
    ranges = {'train': 11000, 'dev': 12000, 'holdout': 13000}
    number = ranges[split] + index
    family = f'new-evidence-{number}'
    scope = f'unit N{number}'
    fields = [('equipment booking allowance', 'hours'), ('courier dispatch quota', 'parcels'),
              ('sample storage interval', 'days'), ('visitor pass duration', 'weeks'),
              ('project compute allocation', 'credits'), ('audit review period', 'months'),
              ('testing container capacity', 'liters'), ('service request limit', 'requests')]
    field, unit = fields[index % len(fields)]
    x, y = f'{17+index*11} {unit}', f'{43+index*13} {unit}'
    if kind == 'altered':
        field, unit = [('minutes per configured hour', 'minutes'), ('days per configured week', 'days'),
                       ('months per configured year', 'months'), ('hours per simulated day', 'hours')][index % 4]
        x, y = f'{37+index*7} {unit}', f'{83+index*11} {unit}'
    count = 1 + index % 3
    facts = [{'scope': f'unit Z{number}-{d}', 'field': field, 'value': f'{99+d} {unit}'}
             for d in range(count)]
    facts += [{'scope': scope, 'field': 'unrelated inventory counter', 'value': f'{71+index} items'}]
    if kind != 'missing':
        facts.append({'scope': scope, 'field': field, 'value': x})
    if kind == 'conflict':
        facts.append({'scope': scope, 'field': field, 'value': y})
    # Document count and placement do not identify the label. Contradiction
    # can occur in one document; irrelevant extra documents occur in all types.
    random.Random(SEED + number).shuffle(facts)
    docs = [[] for _ in range(count)]
    for i, f in enumerate(facts):
        statement = [f"The {f['field']} for {f['scope']} is {f['value']}.",
                     f"{f['scope']}: {f['field']} = {f['value']}.",
                     f"For {f['scope']}, records assign {f['value']} to {f['field']}."][index % 3]
        docs[i % count].append(statement)
    context = '\n'.join(f'[Source {i+1}] ' + ' '.join(lines) for i, lines in enumerate(docs))
    gold = oracle(facts, scope, field)
    assert (kind == 'missing' and gold == 'UNKNOWN') or (kind == 'conflict' and gold == 'CONFLICT') or (kind == 'altered' and gold == x)
    question = [f'What is the recorded {field} for {scope}?',
                f'According to these sources, give the {field} assigned to {scope}.',
                f'Which {field} do the records specify for {scope}?'][index % 3]
    return {'id': f'{family}-{kind}', 'family': family, 'dataset': 'new-evidence-contrasts-v1',
            'split': split, 'case_type': kind, 'context': context, 'question': question,
            'gold': gold, 'aliases': [gold], 'source_facts': facts, 'query_scope': scope,
            'query_field': field, 'document_count': count,
            'source_uri': f'synthetic://evidence-dial-followup/{split}/{index}/{kind}'}


def freeze(root, folder):
    source_path = folder / 'inputs/squad-train-v1.1.json'
    source = json.loads(source_path.read_text())
    assert source['version'] == '1.1'
    destination = folder / 'data'
    if destination.exists():
        raise FileExistsError('Frozen data must never be regenerated after model outputs')
    prior_paths = sorted(p for p in (root / 'artifacts/data').glob('*/*.jsonl')
                         if p.name in ['train.jsonl', 'dev.jsonl', 'holdout.jsonl'])
    prior = [r for p in prior_paths for r in read_jsonl(p)]
    allowed, excluded = real_candidates(source, prior)
    rng = random.Random(SEED)
    rng.shuffle(allowed)
    assert len(allowed) >= 128, f'Only {len(allowed)} disjoint eligible articles; no fallback to overlaps'
    counts = {'train': 32, 'dev': 32, 'holdout': 64}
    articles, rows = {}, {}
    offset = 0
    for split, count in counts.items():
        chosen = allowed[offset:offset+count]; offset += count
        articles[split] = [title for title, _ in chosen]
        rows[split] = []
        for title, candidates in chosen:
            item = dict(rng.choice(candidates))
            item.update(id=f'squad-train-{item["source_id"]}', family='article-'+title,
                        dataset='squad-v1.1-training-article-split', split=split, case_type='answerable',
                        source_uri=f'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json#{item["source_id"]}')
            rows[split].append(item)
        per_type = {'altered': 16, 'missing': 8, 'conflict': 8} if split == 'train' else {k: count for k in ['altered', 'missing', 'conflict']}
        for kind, n in per_type.items():
            rows[split] += [synthetic(split, kind, i) for i in range(n)]
        rng.shuffle(rows[split])
    all_rows = [r for group in rows.values() for r in group]
    assert len({r['id'] for r in all_rows}) == len(all_rows)
    for a, b in [('train', 'dev'), ('train', 'holdout'), ('dev', 'holdout')]:
        assert not set(articles[a]) & set(articles[b])
        assert not {r['family'] for r in rows[a]} & {r['family'] for r in rows[b]}
    old_prompts = {digest(prompt(r)) for r in prior}
    assert all(digest(prompt(r)) not in old_prompts for r in all_rows)
    assert len({digest(prompt(r)) for r in all_rows}) == len(all_rows)
    for split, group in rows.items():
        write_jsonl(destination / f'{split}.jsonl', group)
    write_json(destination / 'manifest.json', {
        'source': 'SQuAD v1.1 official TRAINING split only, plus parsed synthetic contrasts',
        'source_url': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(), 'source_license': 'CC BY-SA 4.0',
        'seed': SEED, 'article_splits': articles, 'prior_data_files': [str(p.relative_to(root)) for p in prior_paths],
        'prior_rows_for_exclusion_only': len(prior), 'excluded_articles': excluded,
        'allowed_article_count': len(allowed), 'overlap_policy': 'Entire article excluded on any prior source ID, normalized question or shared 18-word passage span; split new articles/families before sampling; no model-output filtering.',
        'case_counts': {s: dict(Counter(r['case_type'] for r in group)) for s, group in rows.items()},
        'files': {s: hashlib.sha256((destination/f'{s}.jsonl').read_bytes()).hexdigest() for s in rows},
        'training_known_answer_fraction': .75, 'source_model_calls_before_freeze': 0,
        'new_training_updates_before_freeze': 0, 'holdout_used_for_selection': False,
        'scorer': 'Unchanged frozen protocol.success; supplied answer aliases only; diagnostics separate'})
    return destination
