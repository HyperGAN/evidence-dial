"""Same full-EOS inference/rubric, with one shared first-judgment cache and an explicit run identity.

Coincident answer text across frozen arms reuses its judgment. A file lease
serializes only the short API call, allowing both GPUs to generate concurrently.
"""
import fcntl
import json
from pathlib import Path

from .behavior_fullanswer_generate_v2 import generate, MAX_NEW_TOKENS
from .behavior_protocol_v1 import ask
from .io import write_json, write_jsonl
from .protocol import digest


def evaluate(rows, model, tok, out, resource, *, arm, first_cache, target=None):
    import auditkit as ak
    from auditkit.model import CallableModel
    out, first_cache = Path(out), Path(first_cache)
    first_cache.mkdir(parents=True, exist_ok=True)
    assert not (out / 'summary.json').exists()
    records = []

    def first(question, text):
        with (first_cache / '.calls.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            scores = ask(question, text, first_cache)
        assert scores['measured'], 'Unmeasured judgment; retain committed first'
        return scores

    @ak.scorer(direction=ak.Direction.MAXIMIZE)
    def intended_behavior(sample, output):
        return first(sample.metadata['question'], output)['effect'] / 4

    def call(texts):
        outputs = []
        for text in texts:
            row = next(r for r in rows if r['question'] == text)
            native = generate(model, tok, text, out / 'native' / (row['id'] + '.json'),
                resource, arm=arm, target=target)
            outputs.append(native['output'])
        return outputs

    # AuditKIT's native API does not put experiment_name into RunSpec.run_name.
    # Bind the actual output namespace into model identity to prevent stale
    # cross-run results when an arm label is reused.
    cache_identity = arm + ':' + digest({'out': str(out.resolve()), 'arm': arm})
    callable_model = CallableModel(call, name=cache_identity)
    for row in rows:
        path = out / 'records' / (row['id'] + '.json')
        if path.exists():
            record = json.loads(path.read_text())
            assert record['id'] == row['id'] and record['arm'] == arm and record['question'] == row['question']
            records.append(record)
            continue
        try:
            result = ak.evaluate([ak.Sample(id=row['id'], input=row['question'], metadata=row)],
                model=callable_model, scorers=[intended_behavior],
                experiment_name=digest({'arm': arm, 'out': str(out.resolve()), 'id': row['id']}),
                config=ak.RunConfig(max_tokens=MAX_NEW_TOKENS, temperature=0, max_retries=0, concurrency=1))
        except Exception as error:
            # AuditKIT may raise before returning a RunResult. Keep the exact
            # affected row unmeasured and preserve all native/judge firsts.
            failure = out / ('unmeasured-' + row['id'] + '.json')
            if not failure.exists():
                write_json(failure, {'schema': 'AuditKIT-exception-unmeasured-v5',
                    'id': row['id'], 'arm': arm, 'error_type': type(error).__name__,
                    'error': str(error), 'completed_record_written': False,
                    'native_capture': str(out / 'native' / (row['id'] + '.json')),
                    'automatic_retry': False})
            raise RuntimeError('Native/Jev evaluation unmeasured; preserve firsts') from error
        if result.failed_count or result.errors or len(result.predictions) != 1:
            write_json(out / ('unmeasured-' + row['id'] + '.json'), result.to_dict())
            raise RuntimeError('Native/Jev evaluation unmeasured; preserve firsts')
        native = json.loads((out / 'native' / (row['id'] + '.json')).read_text())
        assert native['output'] == result.predictions[0].raw_output
        scores = first(row['question'], native['output'])
        record = {'id': row['id'], 'case_type': row['case_type'], 'question': row['question'],
            'arm': arm, 'output': native['output'], 'native': native['native'],
            'termination': native['termination'], 'scores': scores,
            'effect': scores['effect'] / 4, 'auditkit': result.to_dict()}
        write_json(path, record)
        records.append(record)
        print(json.dumps({'arm': arm, 'committed': len(records), 'total': len(rows),
            'tokens': len(native['native']['output_ids'])}), flush=True)
    write_jsonl(out / 'predictions.jsonl', records)
    write_json(out / 'summary.json', {'complete': True, 'arm': arm, 'cases': len(records),
        'mean_effect': sum(r['effect'] for r in records) / len(records),
        'mean_sympathy_probability': sum(r['scores']['sympathy'] for r in records) / len(records),
        'mean_sycophancy_probability': sum(r['scores']['sycophancy'] for r in records) / len(records),
        'selector': 'All arms frozen before test inference; no test-based selection',
        'diagnostics_not_selectors': ['sympathy', 'sycophancy']})
    return records
