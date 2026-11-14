"""New separately bound data/target experiments; official shared PG math intact."""
from pathlib import Path
import json
import re
import torch

from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_audio import AudioRow, PreparedAudioResponse, AudioResponseCritic, audio_conditions
from particle_sliders.response import ResponseBackend
from particle_sliders.lifecycle import Binding, atomic_save
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from .binding import Native as OriginalNative, initialize, CAPS, MODEL_IDENTITY
from .benchmark import file_sha
from .curation import accepted, parse_candidate
from .curated_binding import positive_messages
from .io import read_jsonl, write_json
from .protocol import SEED, digest, messages, success

ROOT = Path(__file__).resolve().parents[2]
MODES = ('context', 'trajectory-eos')


def sampled_answer_start(saved, tok, answer):
    """Locate answer tokens in the actual sampled trace, never re-tokenize them."""
    text = saved['text']
    match = re.search(r'\n\s*\*{0,2}(?:FINAL|Answer)\s*:\*{0,2}\s*([^\n]+)\s*$', text, re.I)
    if not match or match[1].strip() != answer:
        raise ValueError('Actual final answer span is not a clean native token suffix')
    offset = match.start(1)
    output = saved['output_ids']
    for i in range(len(output)):
        before = tok.decode(output[:i], skip_special_tokens=True)
        after = tok.decode(output[:i+1], skip_special_tokens=True)
        if len(before) <= offset < len(after):
            if before.rstrip() != text[:offset].rstrip():
                raise ValueError('A sampled token straddles the marker and answer')
            if tok.decode(output[i:], skip_special_tokens=True).strip() != answer:
                raise ValueError('Sampled suffix contains content outside the accepted answer')
            assert output[-1] == 3
            return i
    raise ValueError('No actual sampled answer boundary')


class Native(OriginalNative):
    def __init__(self, model, *, mode, checkpointing=True):
        self.mode = mode
        super().__init__(model, checkpointing=checkpointing)

    def metadata(self):
        value = super().metadata()
        value.update(protocol=f'cohere2-jev-success-followup-{self.mode}-v1',
            target_convention='approved explanation as private positive context; answer-only positions' if self.mode == 'context'
                else 'actual saved unhinted successful trajectory; answer plus actual native turn/EOS positions',
            critic_conditioning='neutral-hidden;both-lengths-div512;history-and-strength-unchanged')
        return value


def build_cache(folder, mode, *, attempt):
    from safetensors.torch import load_file, save_file
    from .evaluate import load_base, tokenizer
    from .followup_resources import FollowupResources
    assert mode in MODES
    out = folder / 'cache' / mode
    selected_path = folder / 'curation/selected.json'
    selected = json.loads(selected_path.read_text())
    assert selected['complete'] and selected['minimum_coverage_passed']
    assert selected['training_source_sha256'] == file_sha(folder/'data/train.jsonl')
    rows = {r['id']: r for r in read_jsonl(folder / 'data/train.jsonl')}
    identity = {'mode': mode, 'selected_sha256': file_sha(selected_path),
                'train_sha256': file_sha(folder/'data/train.jsonl'), 'binding_sha256': file_sha(__file__)}
    if (out/'index.json').exists():
        value = json.loads((out/'index.json').read_text())
        assert value['identity'] == identity and value['responses_sha256'] == file_sha(out/'responses.pt')
        return value
    prepared, index = [], []
    with FollowupResources(1, out/f'resources-{attempt}.json', plan=folder/'plan.json') as resource:
        model = load_base(); tok = tokenizer()
        def frozen_hidden(ids):
            key = digest({'ids': list(ids), 'base': MODEL_IDENTITY})
            path = folder / 'cache/frozen-native' / (key+'.safetensors')
            if path.exists():
                metadata = json.loads(path.with_suffix('.json').read_text())
                assert metadata['ids'] == list(ids) and metadata['sha256'] == file_sha(path)
                return load_file(str(path))['hidden']
            resource.check()
            tensor = torch.tensor([ids], device='cuda:0')
            with torch.inference_mode():
                hidden = model.model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                    use_cache=False, return_dict=True).last_hidden_state[:, -1, :].float().cpu().clone()
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file({'hidden': hidden}, str(path))
            write_json(path.with_suffix('.json'), {'ids': list(ids), 'sha256': file_sha(path)})
            return hidden
        for selection in selected['selection']:
            row = rows[selection['id']]
            assert row['split'] == 'train' and accepted(selection['scores'])
            key = selection['generation_key']
            saved = json.loads((folder/'curation/generations'/f'{key}.json').read_text())
            trajectory = folder/'curation/generations'/f'{key}.safetensors'
            assert saved['text'] == selection['candidate'] and file_sha(trajectory) == selection['trajectory_sha256']
            parsed, error = parse_candidate(row, saved['text'], saved['output_ids'])
            assert error is None and parsed == selection['parsed'] and success(row, parsed['answer'])
            judgment = json.loads((folder/'curation/jev-cache'/(selection['jev_key']+'.json')).read_text())
            assert judgment['http_status'] == 200 and judgment['response']['model'] == 'typesafe/jev-1.14'
            assert judgment['request']['state'] == {'evidence': row['context'], 'question': row['question'],
                                                    'expected_answer': row['gold'], 'candidate': saved['text']}
            assert digest(judgment['response']) == judgment['response_sha256']
            assert judgment['response']['answers']['correct']['noul'] == selection['scores']['correctness']
            assert judgment['response']['answers']['quality']['score'] == selection['scores']['quality']
            neutral_prompt = tok.apply_chat_template(messages(row), add_generation_prompt=True)
            if mode == 'context':
                answer = tok.encode(parsed['answer'], add_special_tokens=False)
                positive = tok.apply_chat_template(positive_messages(row, selection), add_generation_prompt=True)
                states = None; start = None
            else:
                start = sampled_answer_start(saved, tok, parsed['answer'])
                answer = saved['output_ids'][start:]
                positive = saved['input_ids'] + saved['output_ids'][:start]
                states = load_file(str(trajectory))['last_hidden_before_each_sampled_token']
                assert states.shape == (len(saved['output_ids']), 2048)
            for position, target_token in enumerate(answer):
                history = answer[:position]
                nr, pr = tuple(neutral_prompt+history), tuple(positive+history)
                assert len(nr) <= 768 and len(pr) <= 896
                neutral = frozen_hidden(nr)
                target = frozen_hidden(pr) if states is None else states[start+position:start+position+1].clone()
                assert neutral.shape == target.shape == (1, 2048) and torch.isfinite(target).all()
                prepared.append(PreparedAudioResponse(AudioRow(nr, pr, position, 1.), target, neutral))
                index.append({'id': row['id'], 'case_type': row['case_type'], 'position': position,
                    'history_ids': history, 'target_token': target_token, 'neutral_ids': list(nr),
                    'teacher_ids': list(pr), 'generation_key': key, 'jev_key': selection['jev_key'],
                    'target_from_actual_sample': states is not None,
                    'target_sample_position': None if start is None else start+position,
                    'trajectory_sha256': selection['trajectory_sha256']})
            write_json(out/'progress.json', {'cases': len({x['id'] for x in index}), 'positions': len(index)})
    atomic_save(out/'responses.pt', prepared)
    result = {'identity': identity, 'rows': index, 'cases': len(selected['selection']),
              'responses_sha256': file_sha(out/'responses.pt'), 'mode': mode,
              'native_terminal_positions': sum(x['target_token'] in [261001, 3] for x in index),
              'new_test_outputs': 0}
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
    return ResponseBackend('tiny-aya-evidence-dial-followup-'+mode+'-v1', CAPS, prepare, validate)


def inference_backend(mode):
    assert mode in MODES
    return InferenceBackend('tiny-aya-evidence-dial-followup-'+mode+'-v1', CAPS,
                            prepare_inference, validate_inference)
