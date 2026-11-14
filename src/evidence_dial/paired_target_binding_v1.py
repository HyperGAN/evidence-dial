"""Pure PG paired-target critic ablation; training-only privileged conditions.

The native generator response remains its unprojected 2048-coordinate final
hidden state. The official PG game/optimizers/penalty are unchanged. Supplying
the matched TRAIN teacher state to the critic makes pairwise residuals visible;
no teacher context or target is required by the exported inference backend.
"""
import json
from pathlib import Path
import torch
from torch import nn

from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_audio import PreparedAudioResponse, audio_conditions
from particle_sliders.response import ResponseBackend
from particle_sliders.lifecycle import Binding
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from .binding import Native as OriginalNative, initialize, CAPS, MODEL_IDENTITY
from .benchmark import file_sha
from .io import write_json
from .protocol import SEED

NAME = 'tiny-aya-evidence-dial-paired-target-first-answer-v1'
PROTOCOL = 'cohere2-jev-first-answer-paired-target-critic-v1'


class PairedTargetCritic(nn.Module):
    """Expose signed/squared paired residuals and frozen neutral context.

    RMS normalization uses only each fixed teacher vector, equally for real and
    fake. The score is an unbounded learned logit; no CE or reconstruction term
    is added to the generator loss. This is a new architecture, not a proved fix.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6148, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, response, condition):
        neutral, target, metadata = condition[:, :2048], condition[:, 2048:4096], condition[:, 4096:]
        scale = target.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
        residual = (response - target) / scale
        features = torch.cat((residual, residual.square(), neutral / scale, metadata), dim=1)
        return self.net(features).squeeze(-1)


class Native(OriginalNative):
    def __init__(self, model, *, mode, checkpointing=True):
        assert mode == 'first-answer'
        self.mode = mode
        super().__init__(model, checkpointing=checkpointing)

    def metadata(self):
        value = super().metadata()
        value.update(protocol=PROTOCOL,
            target_convention='same retained60 actual successful first-answer hidden targets;59 unhinted,1 reference-assisted;no answer prefix in neutral',
            critic_conditioning='frozen TRAIN neutral and matched teacher hidden states plus normalized lengths,position,strength;paired signed/squared residuals;no inference teacher',
            generator_objective='unchanged official PG adversarial+prior;no CE,no auxiliary reconstruction,no SFT warmstart')
        return value


def critic_conditions(rows):
    original = audio_conditions(rows)
    metadata = original[:, 2048:].clone()
    metadata[:, :2] /= 512.
    return torch.cat((original[:, :2048], torch.cat([r.target.detach() for r in rows]), metadata), dim=1)


def build_cache(folder, mode, *, attempt):
    """The declared CPU-only byte copy reuses the completed target packet."""
    assert mode == 'first-answer'
    cache = folder / 'cache' / mode
    index = json.loads((cache / 'index.json').read_text())
    assert index['cases'] == len(index['rows']) == 60
    assert index['responses_sha256'] == file_sha(cache / 'responses.pt')
    return index


def validate(options):
    if options.get('ema_decay') is not None or options.get('base_precision') != 'bfloat16':
        raise ValueError('Native BF16 and RAW/noEMA required')
    if (options['mode'], options['steps'], options['batch'], options['g_lr']) != ('first-answer', 128, 2, .001):
        raise ValueError('Only separately declared fixed128 paired-target experiment is supported')
    if options['binding_sha256'] != file_sha(__file__):
        raise ValueError('Captured paired-target binding changed')
    if options['cache_index_sha256'] != file_sha(Path(options['cache']) / 'index.json'):
        raise ValueError('Frozen target cache changed')


def prepare(options, *, restored_from=None):
    from .evaluate import load_base
    torch.manual_seed(SEED)
    native = Native(load_base(), mode=options['mode'])
    try:
        critic = PairedTargetCritic().to(device='cuda:0', dtype=torch.float32)
        receipt = Path(options['initialization_receipt'])
        if restored_from is None:
            if receipt.exists():
                raise ValueError('Fresh initialization retained; exactly resume instead')
            value = initialize(native, critic, options['cohort_id'])
            write_json(receipt, {'sha256': value.sha256, 'receipt': value.as_dict()})
        elif not receipt.exists():
            raise ValueError('Resume requires its original scoped initialization receipt')
        cache = Path(options['cache'])
        index = json.loads((cache / 'index.json').read_text())
        assert index['mode'] == 'first-answer' and index['responses_sha256'] == file_sha(cache / 'responses.pt')
        stored = torch.load(cache / 'responses.pt', map_location='cpu', weights_only=False)
        assert len(stored) == index['cases'] == 60
        assert all(x.row.history_tokens == 0 for x in stored)
        rows = [PreparedAudioResponse(x.row, x.target.to('cuda:0'), x.neutral.to('cuda:0')) for x in stored]
        game = ConditionalPG09(native, rows, critic, conditions=critic_conditions(rows), particles=native.particles,
            response_domain='continuous_vector', penalty_input_domain='continuous', batch_size=2,
            seed=SEED, total_steps=128, response_scale=1.,
            recipe_overrides={'lr': .001, 'd_lr_mult': 18., 'prior_lr_mult': 30.})
        assert game.recipe.ema_decay == 0.
        assert sorted(g['lr'] for g in game.opt_g.param_groups) == [.001, .03]
        assert all(abs(g['lr'] - .018) < 1e-12 for g in game.opt_d.param_groups)
        spec = inference_backend(options['mode']).specification(native, model_identity=MODEL_IDENTITY,
            protocol=PROTOCOL, configuration={'mode': 'first-answer', 'rank': 8, 'alpha': 8,
                'bank_shape': [128, 4], 'ema_decay': None})
        return Binding(game, native.close, inference=spec)
    except BaseException:
        native.close()
        raise


def validate_inference(options, spec):
    if options or spec.configuration != {'mode': 'first-answer', 'rank': 8, 'alpha': 8,
                                         'bank_shape': [128, 4], 'ema_decay': None}:
        raise ValueError('Only declared RAW teacher-free inference supported')


def prepare_inference(options, spec):
    from .evaluate import load_base
    native = Native(load_base(), mode='first-answer', checkpointing=False)
    return InferenceBinding(native.network,
        native_identity(native, model_identity=MODEL_IDENTITY, protocol=PROTOCOL),
        native.close, response=native.response)


def response_backend(mode):
    assert mode == 'first-answer'
    return ResponseBackend(NAME, CAPS, prepare, validate)


def inference_backend(mode):
    assert mode == 'first-answer'
    return InferenceBackend(NAME, CAPS, prepare_inference, validate_inference)
