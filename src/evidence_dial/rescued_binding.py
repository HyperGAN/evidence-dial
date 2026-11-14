"""New separately bound data/target experiments; official shared PG math intact."""
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
from .curation import accepted, QUESTIONS
from .io import read_jsonl, write_json
from .protocol import SEED, digest, messages, success

ROOT = Path(__file__).resolve().parents[2]
MODES = ('trajectory-eos',)



class Native(OriginalNative):
    def __init__(self, model, *, mode, checkpointing=True):
        self.mode = mode
        super().__init__(model, checkpointing=checkpointing)

    def metadata(self):
        value = super().metadata()
        value.update(protocol=f'cohere2-jev-rescued-followup-{self.mode}-v1',
            target_convention='per-example reasoning/search procedure and approved working as private positive context; answer-only positions' if self.mode == 'context'
                else 'actual saved per-example reasoning/search trajectory; 59 unhinted plus one labelled reference-assisted; answer plus actual native turn/EOS positions',
            critic_conditioning='neutral-hidden;both-lengths-div512;history-and-strength-unchanged')
        return value


def verify_selection(row, selection, tok):
    """Verify immutable native sample and its original Jev admission, on train only."""
    from .rescue_curation import extract, answer_start
    assert row['split'] == 'train' and row['id'] == selection['id']
    assert accepted(selection['scores'])
    for source in selection['source_files']:
        assert file_sha(ROOT/source['path']) == source['sha256'], source['path']
    saved = json.loads((ROOT/selection['generation_source']).read_text())
    assert digest(saved['identity']) == selection['generation_key']
    assert saved['id'] == row['id'] and saved['text'] == selection['candidate']
    assert saved['identity']['base'] == MODEL_IDENTITY['repo']
    assert saved['identity']['revision'] == MODEL_IDENTITY['revision']
    assert tok.apply_chat_template([{'role':'user','content':saved['identity']['prompt']}], add_generation_prompt=True) == saved['input_ids']
    assert tok.decode(saved['output_ids'], skip_special_tokens=True) == saved['text']
    parsed, error = extract(saved)
    assert error is None and parsed['answer'] == selection['parsed']['answer']
    start = answer_start(saved, tok, parsed)
    if selection.get('native_answer_start') is not None:
        assert start == selection['native_answer_start']
    trajectory = ROOT/selection['trajectory_source']
    assert saved['trajectory_sha256'] == selection['trajectory_sha256'] == file_sha(trajectory)
    judgment = json.loads((ROOT/selection['judgment_source']).read_text())
    assert judgment['request']['questions'] == QUESTIONS
    assert judgment['request']['state'] == {'evidence': row['context'], 'question': row['question'],
                                            'expected_answer': row['gold'], 'candidate': saved['text']}
    assert judgment['http_status'] == 200 and judgment['response']['model'] == 'typesafe/jev-1.14'
    assert digest(judgment['response']) == judgment['response_sha256']
    assert judgment['response']['answers']['correct']['noul'] == selection['scores']['correctness']
    assert judgment['response']['answers']['quality']['score'] == selection['scores']['quality']
    assert judgment['request']['model'] == 'typesafe/jev-1.14'
    assert saved['identity'].get('assistance', {}).get('reference_answer_supplied', False) == selection['reference_answer_supplied']
    return saved, parsed, start


def build_cache(folder, mode, *, attempt):
    from safetensors.torch import load_file, save_file
    from .evaluate import load_base, tokenizer
    from .gpu0_resources import GPU0FollowupResources
    assert mode in MODES
    out = folder/'cache'/mode
    selected_path = folder/'curation/selected.json'
    selected = json.loads(selected_path.read_text())
    assert selected['complete'] and selected['minimum_coverage_passed'] and len(selected['selection']) == 60
    assert selected['training_source_sha256'] == file_sha(folder/'data/train.jsonl')
    rows = {r['id']: r for r in read_jsonl(folder/'data/train.jsonl')}
    identity = {'mode': mode, 'selected_sha256': file_sha(selected_path),
                'train_sha256': file_sha(folder/'data/train.jsonl'), 'binding_sha256': file_sha(__file__)}
    if (out/'index.json').exists():
        value = json.loads((out/'index.json').read_text())
        assert value['identity'] == identity and value['responses_sha256'] == file_sha(out/'responses.pt')
        return value
    tok = tokenizer()
    verified = [(s, *verify_selection(rows[s['id']], s, tok)) for s in selected['selection']]
    prepared, index = [], []
    reused, created = set(), set()
    with GPU0FollowupResources(0, out/f'resources-{attempt}.json', plan=folder/'plan.json') as resource:
        model = load_base()
        def frozen_hidden(ids):
            key = digest({'ids': list(ids), 'base': MODEL_IDENTITY})
            local = folder/'cache/frozen-native'/(key+'.safetensors')
            old = ROOT/'artifacts/rescued-followup-20261111-v2/cache/frozen-native'/(key+'.safetensors')
            path = local if local.exists() else old if old.exists() else local
            if path.exists():
                metadata = json.loads(path.with_suffix('.json').read_text())
                assert metadata['ids'] == list(ids) and metadata['sha256'] == file_sha(path)
                reused.add(str(path.relative_to(ROOT)))
                return load_file(str(path))['hidden']
            resource.check()
            tensor = torch.tensor([ids], device='cuda:0')
            with torch.inference_mode():
                hidden = model.model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                    use_cache=False, return_dict=True).last_hidden_state[:, -1, :].float().cpu().clone()
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file({'hidden': hidden}, str(path))
            write_json(path.with_suffix('.json'), {'ids': list(ids), 'sha256': file_sha(path)})
            created.add(str(path.relative_to(ROOT)))
            return hidden
        for selection, saved, parsed, start in verified:
            row = rows[selection['id']]
            neutral_prompt = tok.apply_chat_template(messages(row), add_generation_prompt=True)
            # Rich working stays private to the positive trajectory. The neutral
            # input is the frozen ordinary question plus the same answer prefix.
            answer = saved['output_ids'][start:]
            positive = saved['input_ids'] + saved['output_ids'][:start]
            states = load_file(str(ROOT/selection['trajectory_source']))['last_hidden_before_each_sampled_token']
            assert states.shape == (len(saved['output_ids']), 2048)
            for position, target_token in enumerate(answer):
                history = answer[:position]
                nr, pr = tuple(neutral_prompt+history), tuple(positive+history)
                assert len(nr) <= 768 and len(pr) <= 1536
                neutral = frozen_hidden(nr)
                target = states[start+position:start+position+1].clone()
                assert neutral.shape == target.shape == (1, 2048)
                assert torch.isfinite(target).all() and torch.isfinite(neutral).all()
                prepared.append(PreparedAudioResponse(AudioRow(nr, pr, position, 1.), target, neutral))
                index.append({'id': row['id'], 'case_type': row['case_type'], 'position': position,
                    'history_ids': history, 'target_token': target_token, 'neutral_ids': list(nr),
                    'teacher_ids': list(pr), 'generation_key': selection['generation_key'],
                    'jev_key': selection['scores']['key'], 'target_from_actual_sample': True,
                    'target_sample_position': start+position,
                    'trajectory_sha256': selection['trajectory_sha256'],
                    'reference_answer_supplied': selection['reference_answer_supplied']})
            write_json(out/'progress.json', {'cases': len({x['id'] for x in index}), 'positions': len(index)})
    atomic_save(out/'responses.pt', prepared)
    result = {'identity': identity, 'rows': index, 'cases': len(selected['selection']),
              'responses_sha256': file_sha(out/'responses.pt'), 'mode': mode,
              'native_terminal_positions': sum(x['target_token'] in [261001, 3] for x in index),
              'frozen_hidden_sources_reused': sorted(reused), 'new_frozen_hidden_sources': sorted(created),
              'new_test_outputs': 0, 'test_scorer_changed': False,
              'canonical_gold_rewritten': False, 'teacher_forced_positive_targets': False}
    write_json(out/'index.json', result)
    return result


def validate(options):
    if options.get('ema_decay') is not None or options.get('base_precision') != 'bfloat16':
        raise ValueError('RAW/noEMA and native BF16 required')
    if options['mode'] not in MODES or (options['steps'], options['batch'], options['g_lr']) != (128, 2, .001):
        raise ValueError('Only the declared fixed128/batch2/G001 experiment is supported')
    if options['binding_sha256'] != file_sha(__file__):
        raise ValueError('Frozen follow-up source differs')
    if options['cache_index_sha256'] != file_sha(Path(options['cache'])/'index.json'):
        raise ValueError('Frozen training-only cache differs')


def prepare(options, *, restored_from=None):
    from .evaluate import load_base
    torch.manual_seed(SEED)
    native = Native(load_base(), mode=options['mode'])
    try:
        critic = AudioResponseCritic(2048, 2052).to(device='cuda:0', dtype=torch.float32)
        receipt = Path(options['initialization_receipt'])
        if restored_from is None:
            if receipt.exists():
                raise ValueError('Fresh initialization already retained; resume instead')
            value = initialize(native, critic, options['cohort_id'])
            write_json(receipt, {'sha256': value.sha256, 'receipt': value.as_dict()})
        elif not receipt.exists():
            raise ValueError('Resume requires its original scoped initialization receipt')
        cache = Path(options['cache'])
        index = json.loads((cache/'index.json').read_text())
        assert index['mode'] == options['mode'] and index['responses_sha256'] == file_sha(cache/'responses.pt')
        stored = torch.load(cache/'responses.pt', map_location='cpu', weights_only=False)
        rows = [PreparedAudioResponse(x.row, x.target.to('cuda:0'), x.neutral.to('cuda:0')) for x in stored]
        conditions = audio_conditions(rows); conditions[:, 2048:2050] /= 512.
        game = ConditionalPG09(native, rows, critic, conditions=conditions, particles=native.particles,
            response_domain='continuous_vector', penalty_input_domain='continuous', batch_size=2,
            seed=SEED, total_steps=128, response_scale=1.,
            recipe_overrides={'lr': .001, 'd_lr_mult': 18., 'prior_lr_mult': 30.})
        assert game.recipe.ema_decay == 0.
        backend = inference_backend(options['mode'])
        spec = backend.specification(native, model_identity=MODEL_IDENTITY,
            protocol=native.metadata()['protocol'], configuration={'mode': options['mode'], 'rank': 8,
                'alpha': 8, 'bank_shape': [128, 4], 'ema_decay': None})
        return Binding(game, native.close, inference=spec)
    except BaseException:
        native.close(); raise


def validate_inference(options, spec):
    if options or spec.configuration != {'mode': spec.configuration.get('mode'), 'rank': 8,
                                        'alpha': 8, 'bank_shape': [128, 4], 'ema_decay': None}:
        raise ValueError('Only declared RAW follow-up inference is supported')
    assert spec.configuration['mode'] in MODES


def prepare_inference(options, spec):
    from .evaluate import load_base
    native = Native(load_base(), mode=spec.configuration['mode'], checkpointing=False)
    return InferenceBinding(native.network,
        native_identity(native, model_identity=MODEL_IDENTITY, protocol=native.metadata()['protocol']),
        native.close, response=native.response)


def response_backend(mode):
    assert mode in MODES
    return ResponseBackend('tiny-aya-evidence-dial-rescued-followup-'+mode+'-v1', CAPS, prepare, validate)


def inference_backend(mode):
    assert mode in MODES
    return InferenceBackend('tiny-aya-evidence-dial-rescued-followup-'+mode+'-v1', CAPS,
                            prepare_inference, validate_inference)
