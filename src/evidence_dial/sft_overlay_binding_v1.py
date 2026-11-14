"""Separate SFT+PG experiment: frozen saved SFT with a fresh zero-output slider.

The existing pure PG cohorts are unchanged. No merge/retraining/reset of the SFT
adapter occurs. OFF keeps its complete PEFT branch. Teacher responses are the
same retained successful base-model TRAIN states; only neutral states must be
recached under the stronger frozen SFT student.
"""
from pathlib import Path
import json
import torch
from torch import nn

from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_audio import ARLastHiddenResponseAdapter, PreparedAudioResponse
from particle_sliders.routed_projection import TinyParticleProjection
from particle_sliders.response import ResponseBackend
from particle_sliders.lifecycle import Binding, atomic_save
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from .binding import Native as BaseNative, Slider, initialize, CAPS, MODEL_IDENTITY
from .paired_target_binding_v1 import PairedTargetCritic, critic_conditions
from .benchmark import file_sha
from .io import write_json
from .protocol import SEED, digest

ROOT = Path(__file__).resolve().parents[2]
NAME = 'tiny-aya-evidence-dial-frozen-sft60-pg-overlay-v1'
PROTOCOL = 'cohere2-frozen-sft60-zero-output-pg-overlay-v1'
SFT = ROOT / 'artifacts/rescued-trajectories-20261111-v1/runs/aligntune/adapter-000128'
SFT_SHA256 = '1d6bf4efc49ca4043a8d6a53119401788eb7b34d6e4461428490e3bc9202f898'
IDENTITY = {**MODEL_IDENTITY, 'frozen_SFT_adapter_sha256': SFT_SHA256,
            'method_scope': 'HybridSFT128+newPG; not pure PG from original pretrained weights'}


def load_frozen_sft():
    from peft import PeftModel
    from .evaluate import load_base
    assert file_sha(SFT / 'adapter_model.safetensors') == SFT_SHA256
    wrapped = PeftModel.from_pretrained(load_base(), str(SFT), is_trainable=False).eval()
    wrapped.requires_grad_(False)
    verify_frozen_sft(wrapped)
    return wrapped, wrapped.get_base_model()


def verify_frozen_sft(wrapped):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    expected = load_file(str(SFT / 'adapter_model.safetensors'))
    actual = get_peft_model_state_dict(wrapped)
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[k].detach().cpu(), expected[k]) for k in expected)
    assert not any(p.requires_grad for p in wrapped.parameters())


def overlay_targets(model):
    """Attach below frozen PEFT additions, without replacing their forwards."""
    from peft.tuners.lora.layer import Linear as PeftLinear
    assert model.config.model_type == 'cohere2' and model.config.hidden_size == 2048
    assert len(model.model.layers) == 36
    targets = {}
    for i, layer in enumerate(model.model.layers):
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            outer = getattr(layer.self_attn, name)
            if not isinstance(outer, PeftLinear) or outer.merged or outer.disable_adapters:
                raise ValueError('Exact active, unmerged frozen PEFT attention branch required')
            host = outer.get_base_layer()
            if type(host) is not nn.Linear or any(p.requires_grad for p in outer.parameters()):
                raise ValueError('Frozen SFT/base Linear required before adding the new slider')
            targets[f'model.layers.{i}.self_attn.{name}.base_layer'] = host
    return targets


class OverlaySlider(Slider):
    def __init__(self, model):
        nn.Module.__init__(self)
        targets = overlay_targets(model)
        self.particles = nn.Parameter(torch.randn(128, 4, device=next(model.parameters()).device))
        self.adapters = nn.ModuleDict()
        self.hosts = []
        try:
            for name, host in targets.items():
                adapter = TinyParticleProjection(name, host, self.particles, rank=8, alpha=8)
                self.hosts.append((host, adapter))
                self.adapters[name.replace('.', '-')] = adapter
            self.to(device=self.particles.device, dtype=torch.float32)
        except BaseException:
            self.close()
            raise


class Native(BaseNative):
    def __init__(self, model, *, checkpointing=True):
        slider = OverlaySlider(model)
        try:
            ARLastHiddenResponseAdapter.__init__(self, model, slider, model_identity=IDENTITY,
                checkpoint_modules=tuple(model.model.layers) if checkpointing else (), penalty_input_domain='continuous')
        except BaseException:
            slider.close()
            raise

    def metadata(self):
        value = super().metadata()
        value.update(protocol=PROTOCOL, model_identity=IDENTITY,
            target_convention='Retained60 actual successful base-model first-answer TRAIN states; neutral recached under frozenSFT60',
            student_conditioning='ordinary evidence/question; frozen SFT branch plus fresh PG slider; no inference teacher',
            critic_conditioning='same paired-target critic and official PG objective as declared pure paired control',
            initialization_scope='Preserve original backbone and saved SFT LoRA; fresh-only zero-output PG adapter/critic/bank law')
        return value


def build_cache(folder, *, resource):
    """Only neutral SFT forwards are new; accepted teacher targets stay exact."""
    from safetensors.torch import save_file, load_file
    source = ROOT / 'artifacts/first-answer-control-20261111-v1/cache/first-answer'
    index = json.loads((source / 'index.json').read_text())
    assert index['responses_sha256'] == file_sha(source / 'responses.pt')
    selected = source.parents[1] / 'curation/selected.json'
    selection = json.loads(selected.read_text())
    assert index['identity']['selected_sha256'] == file_sha(selected)
    assert selection['training_source_sha256'] == file_sha(folder / 'data/train.jsonl')
    assert {i['id'] for i in index['rows']} == {s['id'] for s in selection['selection']}
    rows = torch.load(source / 'responses.pt', map_location='cpu', weights_only=False)
    assert len(rows) == index['cases'] == 60 and all(x.row.history_tokens == 0 for x in rows)
    destination = folder / 'cache'
    if (destination / 'index.json').exists():
        raise FileExistsError('Reuse the saved overlay neutral cache; never recache automatically')
    wrapped, model = None, None
    prepared = []
    created = reused = 0
    try:
        with torch.inference_mode():
            for row in rows:
                resource.check()
                key = digest({'frozen_SFT_sha256': SFT_SHA256, 'base': MODEL_IDENTITY, 'input_ids': row.row.neutral_ids})
                path = destination / 'frozen-neutral' / (key + '.safetensors')
                if path.exists():
                    metadata = json.loads(path.with_suffix('.json').read_text())
                    assert metadata['ids'] == list(row.row.neutral_ids) and metadata['sha256'] == file_sha(path)
                    hidden = load_file(str(path))['hidden']
                    reused += 1
                else:
                    if model is None:
                        wrapped, model = load_frozen_sft()
                    tokens = torch.tensor([row.row.neutral_ids], device='cuda:0')
                    hidden = model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False,
                        return_dict=True).last_hidden_state[:, -1, :].float().cpu().clone()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_file({'hidden': hidden}, str(path))
                    write_json(path.with_suffix('.json'), {'ids': list(row.row.neutral_ids), 'sha256': file_sha(path),
                        'frozen_SFT_sha256': SFT_SHA256})
                    created += 1
                prepared.append(PreparedAudioResponse(row.row, row.target.clone(), hidden))
    finally:
        del model, wrapped
    atomic_save(destination / 'responses.pt', prepared)
    result = {'cases': 60, 'rows': index['rows'],
        'parent_index_sha256': file_sha(source / 'index.json'), 'parent_responses_sha256': file_sha(source / 'responses.pt'),
        'responses_sha256': file_sha(destination / 'responses.pt'), 'frozen_SFT_adapter_sha256': SFT_SHA256,
        'teacher_targets_bit_equal_parent': all(torch.equal(a.target, b.target) for a, b in zip(rows, prepared)),
        'new_teacher_or_mining_calls': 0, 'new_neutral_SFT_forwards_this_attempt': created,
        'reused_neutral_SFT_forwards': reused, 'new_training_updates': 0}
    write_json(destination / 'index.json', result)
    return result


def validate(options):
    if (options.get('ema_decay') is not None or options.get('base_precision') != 'bfloat16'
            or (options['steps'], options['batch'], options['g_lr']) != (128, 2, .001)):
        raise ValueError('Separate fixed128 RAW/noEMA SFT+PG experiment required')
    assert options['binding_sha256'] == file_sha(__file__)
    assert options['cache_index_sha256'] == file_sha(Path(options['cache']) / 'index.json')


def prepare(options, *, restored_from=None):
    torch.manual_seed(SEED)
    wrapped, model = load_frozen_sft()
    native = Native(model)
    try:
        critic = PairedTargetCritic().to(device='cuda:0', dtype=torch.float32)
        receipt = Path(options['initialization_receipt'])
        if restored_from is None:
            if receipt.exists():
                raise ValueError('Fresh initialization already saved; exactly resume')
            value = initialize(native, critic, options['cohort_id'])
            write_json(receipt, {'sha256': value.sha256, 'receipt': value.as_dict()})
        elif not receipt.exists():
            raise ValueError('Original fresh-only initialization receipt required')
        verify_frozen_sft(wrapped)
        cache = Path(options['cache'])
        index = json.loads((cache / 'index.json').read_text())
        assert index['responses_sha256'] == file_sha(cache / 'responses.pt')
        assert index['frozen_SFT_adapter_sha256'] == SFT_SHA256
        stored = torch.load(cache / 'responses.pt', map_location='cpu', weights_only=False)
        rows = [PreparedAudioResponse(x.row, x.target.to('cuda:0'), x.neutral.to('cuda:0')) for x in stored]
        game = ConditionalPG09(native, rows, critic, conditions=critic_conditions(rows), particles=native.particles,
            response_domain='continuous_vector', penalty_input_domain='continuous', batch_size=2,
            seed=SEED, total_steps=128, response_scale=1.,
            recipe_overrides={'lr': .001, 'd_lr_mult': 18., 'prior_lr_mult': 30.})
        assert game.recipe.ema_decay == 0.
        spec = INFERENCE.specification(native, model_identity=IDENTITY, protocol=PROTOCOL,
            configuration={'rank': 8, 'alpha': 8, 'bank_shape': [128, 4], 'ema_decay': None,
                           'frozen_SFT_adapter_sha256': SFT_SHA256})
        return Binding(game, native.close, inference=spec)
    except BaseException:
        native.close()
        raise


def validate_inference(options, spec):
    if options or spec.configuration != {'rank': 8, 'alpha': 8, 'bank_shape': [128, 4], 'ema_decay': None,
                                         'frozen_SFT_adapter_sha256': SFT_SHA256}:
        raise ValueError('Only declared frozenSFT+RAW PG inference supported')


def prepare_inference(options, spec):
    wrapped, model = load_frozen_sft()
    native = Native(model, checkpointing=False)
    return InferenceBinding(native.network, native_identity(native, model_identity=IDENTITY, protocol=PROTOCOL),
                            native.close, response=native.response)


BACKEND = ResponseBackend(NAME, CAPS, prepare, validate)
INFERENCE = InferenceBackend(NAME, CAPS, prepare_inference, validate_inference)


def response_backend(mode):
    assert mode == 'first-answer'
    return BACKEND


def inference_backend(mode):
    assert mode == 'first-answer'
    return INFERENCE
