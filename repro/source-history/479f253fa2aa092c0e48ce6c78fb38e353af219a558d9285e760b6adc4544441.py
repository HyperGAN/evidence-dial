"""Jev-curated reachable-response targets; separate shared-core registration."""
from pathlib import Path
import json
import torch

from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_audio import AudioRow, PreparedAudioResponse, AudioResponseCritic, audio_conditions
from particle_sliders.response import ResponseBackend
from particle_sliders.lifecycle import Binding, atomic_save
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity

from .binding import Native as OriginalNative, initialize, CAPS, MODEL_IDENTITY
from .benchmark import file_sha
from .protocol import SEED, messages, digest, success
from .io import read_jsonl, write_json
from .curation import accepted, parse_candidate, QUESTIONS

ROOT = Path(__file__).resolve().parents[2]
NAME = 'tiny-aya-evidence-dial-curated-v1'
PROTOCOL = 'cohere2-jev-curated-success-paired-native-last-hidden-v1'


class Native(OriginalNative):
    def metadata(self):
        value = super().metadata()
        value.update(protocol=PROTOCOL,
            target_convention='Jev-approved Tiny Aya explanation as positive context; exact same evidence/question and accepted-answer prefix',
            positive_source='unhinted frozen-base successful sample; exact trajectory/seed/Jev provenance retained')
        return value


def positive_messages(row, selected):
    text = messages(row)[0]['content']
    assert text.endswith('Answer:')
    return [{'role': 'user', 'content': text[:-len('Answer:')] +
             'Verified successful Tiny Aya response (private training context):\n' + selected['candidate'] + '\nAnswer:'}]


def build_cache(curation, out, *, attempt='first'):
    """Persist each exact pair before assembling the shared-core target packet."""
    from safetensors.torch import load_file, save_file
    from .evaluate import load_base, tokenizer
    from .resources import Resources
    selected_file = curation / 'selected.json'
    selected = json.loads(selected_file.read_text())
    assert selected['complete'] and selected['source_cases'] == 64 and selected['minimum_coverage_passed']
    source_rows = read_jsonl(ROOT / 'artifacts/data/policy-v1/train.jsonl')
    by_id = {r['id']: r for r in selected['selection']}
    assert len(source_rows) == 64 and set(by_id).issubset({r['id'] for r in source_rows})
    rows = [r for r in source_rows if r['id'] in by_id]
    assert len(rows) == len(by_id) == selected['cases']
    identity = {'protocol': PROTOCOL, 'selected_sha256': file_sha(selected_file),
                'training_sha256': file_sha(ROOT / 'artifacts/data/policy-v1/train.jsonl'),
                'curated_binding_sha256': file_sha(__file__), 'question_sha256': digest(QUESTIONS)}
    if (out / 'index.json').exists():
        existing = json.loads((out / 'index.json').read_text())
        assert existing['identity'] == identity and existing['responses_sha256'] == file_sha(out / 'responses.pt')
        return existing
    if (out / 'identity.json').exists():
        assert json.loads((out / 'identity.json').read_text()) == identity
    else:
        write_json(out / 'identity.json', identity)
    prepared, index = [], []
    with Resources(1, out / f'pair-cache-resources-{attempt}.json') as resource:
        model = load_base(); tok = tokenizer()
        with torch.inference_mode():
            for row in rows:
                item = by_id[row['id']]
                assert accepted(item['scores']) and success(row, item['parsed']['answer'])
                generation = json.loads((curation / 'generations' / (item['generation_key']+'.json')).read_text())
                assert generation['id'] == row['id'] and generation['text'] == item['candidate']
                assert file_sha(curation / 'generations' / (item['generation_key']+'.safetensors')) == item['trajectory_sha256']
                parsed, error = parse_candidate(row, generation['text'], generation['output_ids'])
                assert error is None and parsed == item['parsed']
                judgment = json.loads((curation / 'jev-cache' / (item['jev_key']+'.json')).read_text())
                assert judgment['http_status'] == 200 and judgment['request']['state']['candidate'] == item['candidate']
                assert judgment['request']['state'] == {'evidence': row['context'], 'question': row['question'],
                                                        'expected_answer': row['gold'], 'candidate': item['candidate']}
                assert digest(judgment['response']) == judgment['response_sha256']
                assert judgment['request']['questions'] == QUESTIONS
                actual = judgment['response']['answers']
                assert judgment['response']['model'] == 'typesafe/jev-1.14'
                assert actual['correct']['noul'] == item['scores']['correctness']
                assert actual['quality']['score'] == item['scores']['quality']
                answer = tok.encode(item['parsed']['answer'], add_special_tokens=False)
                neutral = tok.apply_chat_template(messages(row), add_generation_prompt=True)
                positive = tok.apply_chat_template(positive_messages(row, item), add_generation_prompt=True)
                for position in range(len(answer)):
                    resource.check(); history = tuple(answer[:position])
                    nr, pr = tuple(neutral)+history, tuple(positive)+history
                    assert len(nr) <= 512 and len(pr) <= 768
                    provenance = {'id': row['id'], 'position': position, 'history_ids': list(history),
                        'neutral_ids': list(nr), 'teacher_ids': list(pr), 'target_token': answer[position],
                        'generation_key': item['generation_key'], 'jev_key': item['jev_key'],
                        'generation_seed': item['seed'], 'trajectory_sha256': item['trajectory_sha256']}
                    key = digest({'model': MODEL_IDENTITY, 'protocol': PROTOCOL, 'row': provenance})
                    file = out / 'pairs' / (key+'.safetensors'); meta = file.with_suffix('.json')
                    if meta.exists():
                        record = json.loads(meta.read_text())
                        assert record['provenance'] == provenance and record['sha256'] == file_sha(file)
                        pair = load_file(str(file))
                    else:
                        if file.exists():raise RuntimeError('Orphan pair retained; reconcile before recomputing')
                        pair = {}
                        for name, ids in [('neutral', nr), ('target', pr)]:
                            ids_tensor = torch.tensor([ids], device='cuda:0')
                            pair[name] = model.model(input_ids=ids_tensor, attention_mask=torch.ones_like(ids_tensor),
                                use_cache=False, return_dict=True).last_hidden_state[:, -1, :].float().cpu().clone()
                        file.parent.mkdir(parents=True, exist_ok=True)
                        save_file(pair, str(file)); write_json(meta, {'provenance': provenance, 'sha256': file_sha(file)})
                    assert all(p.shape == (1, 2048) and torch.isfinite(p).all() for p in pair.values())
                    prepared.append(PreparedAudioResponse(AudioRow(nr, pr, position, 1.), pair['target'], pair['neutral']))
                    index.append({**provenance, 'pair_key': key, 'pair_sha256': file_sha(file)})
                write_json(out / 'progress.json', {'cached_cases': len({r['id'] for r in index}), 'positions': len(index)})
    atomic_save(out / 'responses.pt', prepared)
    result = {'protocol': PROTOCOL, 'identity': identity, 'cases': len(rows), 'source_cases': 64,
              'ineligible_cases': selected['ineligible_cases'], 'coverage': selected['coverage'], 'rows': index,
              'responses_sha256': file_sha(out / 'responses.pt'), 'teacher_information_evaluated': False,
              'cache_provenance': 'Every target links to an accepted native sample and actual Jev decision'}
    write_json(out / 'index.json', result)
    return result


def validate(options):
    if (options.get('ema_decay') is not None or options.get('base_precision') != 'bfloat16'
            or options['batch'] != 2 or options['steps'] != 128 or options['g_lr'] != .001):
        raise ValueError('Frozen new G.001/128/batch2/RAW recipe required')
    if options['curated_binding_sha256'] != file_sha(__file__):
        raise ValueError('Captured curated source differs; preserve state and bind an explicit derivative')
    cache = Path(options['cache'])
    if options['cache_index_sha256'] != file_sha(cache / 'index.json'):
        raise ValueError('Frozen successful-sample target cache changed')


def prepare(options, *, restored_from=None):
    from .evaluate import load_base
    torch.manual_seed(SEED); native = Native(load_base())
    try:
        critic = AudioResponseCritic(2048, 2052).to(device='cuda:0', dtype=torch.float32)
        receipt = Path(options['initialization_receipt'])
        if restored_from is None:
            if receipt.exists():raise ValueError('Fresh receipt exists; resume instead of resetting it')
            value = initialize(native, critic, options['cohort_id'])
            write_json(receipt, {'sha256': value.sha256, 'receipt': value.as_dict()})
        elif not receipt.is_file():raise ValueError('Resume needs its original initialization receipt')
        cache = Path(options['cache']); index = json.loads((cache / 'index.json').read_text())
        assert index['protocol'] == PROTOCOL and index['responses_sha256'] == file_sha(cache / 'responses.pt')
        original = torch.load(cache / 'responses.pt', map_location='cpu', weights_only=False)
        rows = [PreparedAudioResponse(r.row, r.target.to('cuda:0'), r.neutral.to('cuda:0')) for r in original]
        game = ConditionalPG09(native, rows, critic, conditions=audio_conditions(rows), particles=native.particles,
            response_domain='continuous_vector', penalty_input_domain='continuous', batch_size=2,
            seed=SEED, total_steps=128, response_scale=1.,
            recipe_overrides={'lr': .001, 'd_lr_mult': 18., 'prior_lr_mult': 30.})
        assert sorted(g['lr'] for g in game.opt_g.param_groups) == [.001, .03]
        assert all(abs(g['lr']-.018) < 1e-12 for g in game.opt_d.param_groups)
        assert game.recipe.ema_decay == 0.
        spec = INFERENCE.specification(native, model_identity=MODEL_IDENTITY, protocol=PROTOCOL,
            configuration={'rank': 8, 'alpha': 8, 'bank_shape': [128, 4], 'ema_decay': None})
        return Binding(game, native.close, inference=spec)
    except BaseException:
        native.close(); raise


def validate_inference(options, spec):
    if options or spec.configuration != {'rank': 8, 'alpha': 8, 'bank_shape': [128, 4], 'ema_decay': None}:
        raise ValueError('Only pinned native RAW inference is supported')


def prepare_inference(options, spec):
    from .evaluate import load_base
    native = Native(load_base(), checkpointing=False)
    return InferenceBinding(native.network, native_identity(native, model_identity=MODEL_IDENTITY, protocol=PROTOCOL),
                            native.close, response=native.response)


BACKEND = ResponseBackend(NAME, CAPS, prepare, validate)
INFERENCE = InferenceBackend(NAME, CAPS, prepare_inference, validate_inference)
