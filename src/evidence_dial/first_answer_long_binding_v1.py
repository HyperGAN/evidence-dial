"""Longer fixed-horizon hyperparameter cohorts of the original ParticleGAN game.

Reuse the original native adapter, critic class, conditioning, targets, official
loss and optimizers. Only the declared horizon and learning rates vary. Completed
128-step cohorts are immutable; these are fresh 768-step schedule identities.
"""
import json
from pathlib import Path
import torch

from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_audio import PreparedAudioResponse, AudioResponseCritic, audio_conditions
from particle_sliders.response import ResponseBackend
from particle_sliders.lifecycle import Binding
from .binding import initialize, CAPS, MODEL_IDENTITY
from .first_answer_binding_v2 import Native, inference_backend
from .benchmark import file_sha
from .io import write_json
from .protocol import SEED

STEPS = 768
RATES = (.00025, .001)


def validate(options):
    if options.get('ema_decay') is not None or options.get('base_precision') != 'bfloat16':
        raise ValueError('RAW/noEMA and native BF16 required')
    if (options['mode'], options['steps'], options['batch']) != ('first-answer', STEPS, 2):
        raise ValueError('Only the declared first-answer768/batch2 cohorts are supported')
    if options['g_lr'] not in RATES or options['binding_sha256'] != file_sha(__file__):
        raise ValueError('Undeclared rate or changed binding')
    cache = Path(options['cache'])
    if options['cache_index_sha256'] != file_sha(cache / 'index.json'):
        raise ValueError('Frozen original first-answer targets differ')
    index = json.loads((cache / 'index.json').read_text())
    if index['cases'] != 60 or len(index['rows']) != 60 or index['native_terminal_positions'] != 0:
        raise ValueError('Expected the same60 first-answer training targets')
    if index['responses_sha256'] != file_sha(cache / 'responses.pt'):
        raise ValueError('Frozen target tensor bytes differ')


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
        stored = torch.load(Path(options['cache']) / 'responses.pt', map_location='cpu', weights_only=False)
        rows = [PreparedAudioResponse(x.row, x.target.to('cuda:0'), x.neutral.to('cuda:0')) for x in stored]
        conditions = audio_conditions(rows)
        conditions[:, 2048:2050] /= 512.
        game = ConditionalPG09(native, rows, critic, conditions=conditions, particles=native.particles,
            response_domain='continuous_vector', penalty_input_domain='continuous', batch_size=2,
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


def response_backend(mode):
    assert mode == 'first-answer'
    return ResponseBackend('tiny-aya-evidence-dial-first-answer-long-v1', CAPS, prepare, validate)
