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
MODES = ('first-answer',)



class Native(OriginalNative):
    def __init__(self, model, *, mode, checkpointing=True):
        self.mode = mode
        super().__init__(model, checkpointing=checkpointing)

    def metadata(self):
        value = super().metadata()
        value.update(protocol=f'cohere2-jev-first-answer-followup-{self.mode}-v1',
            target_convention='per-example reasoning/search procedure and approved working as private positive context; answer-only positions' if self.mode == 'context'
                else 'actual sampled first-answer hidden state only; one target per train case; no neutral answer prefix or terminal target; same59 unhinted plus one labelled reference-assisted',
            critic_conditioning='neutral-hidden;both-lengths-div512;history-and-strength-unchanged')
        return value


def build_cache(folder, mode, *, attempt):
    """Reuse verified sampled states; one first-answer target per training case."""
    assert mode in MODES
    parent = ROOT/'artifacts/rescued-trajectories-20261111-v1/cache/trajectory-eos'
    parent_index = json.loads((parent/'index.json').read_text())
    assert parent_index['responses_sha256'] == file_sha(parent/'responses.pt')
    selected_path = folder/'curation/selected.json'
    selected = json.loads(selected_path.read_text())
    assert selected['complete'] and len(selected['selection']) == 60
    assert selected['training_source_sha256'] == file_sha(folder/'data/train.jsonl')
    parent_selected = ROOT/'artifacts/rescued-trajectories-20261111-v1/curation/selected.json'
    assert file_sha(parent_selected) == file_sha(selected_path)
    assert parent_index['identity']['selected_sha256'] == file_sha(parent_selected)
    assert parent_index['identity']['train_sha256'] == file_sha(folder/'data/train.jsonl')
    identity = {'mode':mode, 'selected_sha256':file_sha(selected_path),
        'train_sha256':file_sha(folder/'data/train.jsonl'), 'binding_sha256':file_sha(__file__),
        'parent_index_sha256':file_sha(parent/'index.json'),
        'parent_responses_sha256':file_sha(parent/'responses.pt')}
    out = folder/'cache'/mode
    if (out/'index.json').exists():
        value = json.loads((out/'index.json').read_text())
        assert value['identity'] == identity and value['responses_sha256'] == file_sha(out/'responses.pt')
        return value
    original = torch.load(parent/'responses.pt', map_location='cpu', weights_only=False)
    assert len(original) == len(parent_index['rows'])
    prepared, index = [], []
    for offset, (response, item) in enumerate(zip(original, parent_index['rows'])):
        if item['position'] != 0:
            continue
        assert item['history_ids'] == [] and item['target_from_actual_sample']
        assert response.row.position == 0
        assert tuple(response.row.neutral_ids) == tuple(item['neutral_ids'])
        assert tuple(response.row.teacher_ids) == tuple(item['teacher_ids'])
        assert item['target_token'] not in [261001, 3]
        assert response.neutral.shape == response.target.shape == (1,2048)
        assert torch.isfinite(response.neutral).all() and torch.isfinite(response.target).all()
        prepared.append(response)
        index.append({**item,'parent_response_row':offset,
            'target_convention':'actual sampled first-answer state only; no answer prefix or terminal target'})
    assert len(prepared) == len({i['id']for i in index}) == 60
    assert {i['id']for i in index} == {s['id']for s in selected['selection']}
    atomic_save(out/'responses.pt',prepared)
    value = {'identity':identity, 'rows':index, 'cases':60, 'mode':mode,
        'responses_sha256':file_sha(out/'responses.pt'), 'native_terminal_positions':0,
        'response_targets_recomputed':0, 'new_test_outputs':0, 'case_balanced_targets':True,
        'changes_to_shared_game_optimizer_or_saved_states':False}
    write_json(out/'index.json',value)
    return value


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
    return ResponseBackend('tiny-aya-evidence-dial-first-answer-followup-'+mode+'-v1', CAPS, prepare, validate)


def inference_backend(mode):
    assert mode in MODES
    return InferenceBackend('tiny-aya-evidence-dial-first-answer-followup-'+mode+'-v1', CAPS,
                            prepare_inference, validate_inference)
