"""Original PG128/batch2 on cleaner canonical targets for the original60 IDs.

Native forwards and the original game/critic/optimizer are inherited unchanged.
Positive targets were generated on CPU; neutral states retain original CUDA
values. Both placements and TRAIN-gold-informed curation remain explicit.
"""
import json
from pathlib import Path
import torch

from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_audio import PreparedAudioResponse, AudioResponseCritic, audio_conditions
from particle_sliders.response import ResponseBackend
from particle_sliders.lifecycle import Binding
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from .binding import initialize, CAPS, MODEL_IDENTITY
from .first_answer_binding_v2 import Native as FirstNative
from .benchmark import file_sha
from .io import read_jsonl, write_json
from .protocol import SEED, normalize, success

ROOT = Path(__file__).resolve().parents[2]
NAME = 'tiny-aya-evidence-dial-canonical60-first-v1'
STEPS, BATCH = 128, 2
TRAIN_SHA = '04ff07e205905e2f6343a2d990635b53aab601ff25e30a3b9bb4ee0153ff6636'
AUDIT_SHA = 'fc8c005cb157b5db982da7edf61ddd1b39c18a24f847b57f913e6840b095ed5c'


class Native(FirstNative):
    def metadata(self):
        value = super().metadata()
        value.update(protocol='cohere2-canonical60-first-v1',
            target_convention='actual sampled first-answer hidden state in explicitly canonical-TRAIN-gold-informed private context; CPU4threads/nativeBF16/sdpa teachers and retained original CUDA BF16 ordinary neutral states; same original60 TRAIN IDs, no answer history or terminal target; no CPU/CUDA bitwise equivalence claim')
        return value


def validate(options):
    if options.get('ema_decay') is not None or options.get('base_precision') != 'bfloat16':
        raise ValueError('RAW/noEMA and native BF16 required')
    if (options['mode'], options['steps'], options['batch'], options['g_lr']) != ('first-answer', STEPS, BATCH, .001):
        raise ValueError('Only the declared original PG128/batch2/G001 cohort is supported')
    if options['binding_sha256'] != file_sha(__file__):
        raise ValueError('Frozen canonical60 binding differs')
    data, cache, audit_path = Path(options['training_data']), Path(options['cache']), Path(options['cache_audit'])
    assert options['training_data_sha256'] == file_sha(data) == TRAIN_SHA
    assert options['cache_audit_sha256'] == file_sha(audit_path) == AUDIT_SHA
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'complete60-native-canonical-TRAIN-data-audit' and audit['cases'] == 60
    assert audit['same_case_IDs_and_original_neutral_tensor_values_exact']
    assert audit['all_selected_native_EOS_and_strict_canonical_gold']
    assert audit['all_target_states_exact_before_actual_first_answer_token']
    assert audit['new_model_calls'] == audit['new_PG_or_SFT_updates'] == audit['test_rows_used'] == 0
    assert options['cache_index_sha256'] == file_sha(cache/'index.json') == audit['cache_index_sha256']
    index = json.loads((cache/'index.json').read_text())
    assert index['responses_sha256'] == file_sha(cache/'responses.pt') == audit['cache_responses_sha256']
    assert index['mode'] == 'canonical-recovered-first60' and index['cases'] == len(index['rows']) == 60
    assert index['native_terminal_positions'] == index['neutral_answer_history_tokens'] == 0
    assert index['teacher_placement'] == 'CPU4threads/nativeBF16/sdpa'
    assert index['neutral_placement'] == 'retained original CUDA BF16'
    assert index['CPU_CUDA_bitwise_equivalence_claimed'] is False
    rows = read_jsonl(data)
    assert len(rows) == len({r['id'] for r in rows}) == 64 and all(r['split'] == 'train' for r in rows)
    by_id = {r['id']: r for r in rows}
    original = json.loads((ROOT/'artifacts/first-answer-control-20261111-v1/cache/first-answer/index.json').read_text())
    assert [r['id'] for r in index['rows']] == [r['id'] for r in original['rows']]
    assert len({r['id'] for r in index['rows']}) == 60
    for item, old in zip(index['rows'], original['rows']):
        row = by_id[item['id']]
        assert item['case_type'] == row['case_type'] and item['history_ids'] == [] and item['position'] == 0
        assert item['neutral_ids'] == old['neutral_ids']
        assert item['target_from_actual_sample'] and item['canonical_TRAIN_gold_supplied']
        assert item['target_token'] not in [261001, 3]
        assert normalize(item['teacher_output']) == normalize(row['gold']) and success(row, item['teacher_output'])
        assert file_sha(ROOT/item['native_teacher_source']) == item['native_teacher_sha256']


def prepare(options, *, restored_from=None):
    from .evaluate import load_base
    torch.manual_seed(SEED)
    native = Native(load_base(), mode='first-answer')
    try:
        critic = AudioResponseCritic(2048, 2052).to(device='cuda:0', dtype=torch.float32)
        receipt = Path(options['initialization_receipt'])
        if restored_from is None:
            if receipt.exists():
                raise ValueError('Fresh state already initialized; restore its saved state')
            value = initialize(native, critic, options['cohort_id'])
            write_json(receipt, {'sha256': value.sha256, 'receipt': value.as_dict()})
        elif not receipt.exists():
            raise ValueError('Exact restore requires its original initialization receipt')
        stored = torch.load(Path(options['cache'])/'responses.pt', map_location='cpu', weights_only=False)
        assert len(stored) == 60
        rows = [PreparedAudioResponse(x.row, x.target.to('cuda:0'), x.neutral.to('cuda:0')) for x in stored]
        conditions = audio_conditions(rows)
        conditions[:, 2048:2050] /= 512.
        game = ConditionalPG09(native, rows, critic, conditions=conditions, particles=native.particles,
            response_domain='continuous_vector', penalty_input_domain='continuous', batch_size=BATCH,
            seed=SEED, total_steps=STEPS, response_scale=1.,
            recipe_overrides={'lr': options['g_lr'], 'd_lr_mult': 18., 'prior_lr_mult': 30.})
        assert game.recipe.ema_decay == 0.
        spec = inference_backend('first-answer').specification(native, model_identity=MODEL_IDENTITY,
            protocol=native.metadata()['protocol'], configuration={'mode': 'first-answer', 'rank': 8,
                'alpha': 8, 'bank_shape': [128, 4], 'ema_decay': None})
        return Binding(game, native.close, inference=spec)
    except BaseException:
        native.close()
        raise


def validate_inference(options, spec):
    if options or spec.configuration != {'mode': 'first-answer', 'rank': 8, 'alpha': 8, 'bank_shape': [128, 4], 'ema_decay': None}:
        raise ValueError('Only declared RAW canonical60 inference is supported')


def prepare_inference(options, spec):
    from .evaluate import load_base
    native = Native(load_base(), mode='first-answer', checkpointing=False)
    return InferenceBinding(native.network,
        native_identity(native, model_identity=MODEL_IDENTITY, protocol=native.metadata()['protocol']),
        native.close, response=native.response)


def response_backend(mode):
    assert mode == 'first-answer'
    return ResponseBackend(NAME, CAPS, prepare, validate)


def inference_backend(mode):
    assert mode == 'first-answer'
    return InferenceBackend(NAME, CAPS, prepare_inference, validate_inference)
