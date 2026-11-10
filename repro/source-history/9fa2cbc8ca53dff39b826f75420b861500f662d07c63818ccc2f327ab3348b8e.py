"""Cohere2 native response/inference binding around the shared PG lifecycle.

All game/optimizer/checkpoint mathematics and routed projections are imported
from particle-sliders-core. No Qwen paths, weight EMA or surrogate response.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from particle_sliders.pg09 import ConditionalPG09, PG_SOURCE_PROFILE, PG_VERSION
from particle_sliders.pg09_audio import ARLastHiddenResponseAdapter, AudioRow, PreparedAudioResponse, AudioResponseCritic, audio_conditions
from particle_sliders.routed_projection import TinyParticleProjection
from particle_sliders.response import ResponseBackend, ResponseCapabilities
from particle_sliders.lifecycle import Binding, atomic_save
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity

from .protocol import MODEL, REVISION, SEED, digest, messages
from .io import read_jsonl, write_json

NAME = "tiny-aya-evidence-dial"
PROTOCOL = "cohere2-grounding-matched-gold-prefix-native-last-hidden-v1"
MODEL_IDENTITY = {"repo": MODEL, "revision": REVISION, "hidden_width": 2048, "base_precision": "bfloat16"}
CAPS = ResponseCapabilities("routed_particle_gan", "text_hidden", "native_final_hidden",
    "continuous_vector", (1, 2048), PG_VERSION, PG_SOURCE_PROFILE, "continuous",
    strength_max=None, base_precisions=("bfloat16",),
    qualification="Evidence Dial native checks recorded separately; answer quality requires finished-output evaluation")


def targets(model):
    if model.config.model_type != "cohere2" or model.config.hidden_size != 2048 or len(model.model.layers) != 36:
        raise ValueError("This binding requires the exact Tiny Aya Cohere2 architecture")
    result = {}
    for i, layer in enumerate(model.model.layers):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module = getattr(layer.self_attn, name)
            if type(module) is not nn.Linear or isinstance(getattr(module.forward, "__self__", None), TinyParticleProjection):
                raise ValueError("Unwrapped native Cohere2 attention Linear required")
            result[f"model.layers.{i}.self_attn.{name}"] = module
    return result


class Slider(nn.Module):
    def __init__(self, model):
        super().__init__()
        native_targets = targets(model)
        model.eval().requires_grad_(False)
        self.particles = nn.Parameter(torch.randn(128, 4, device=next(model.parameters()).device))
        self.adapters = nn.ModuleDict()
        self.hosts = []
        try:
            for name, host in native_targets.items():
                adapter = TinyParticleProjection(name, host, self.particles, rank=8, alpha=8)
                self.hosts.append((host, adapter))
                self.adapters[name.replace(".", "-")] = adapter
            self.to(device=self.particles.device, dtype=torch.float32)
        except BaseException:
            self.close()
            raise

    @contextmanager
    def scaled(self, strength):
        strength = float(strength)
        if not math.isfinite(strength):
            raise ValueError("Finite strength required")
        old = [a.multiplier for a in self.adapters.values()]
        try:
            for a in self.adapters.values():
                a.multiplier = strength
            yield self
        finally:
            for a, value in zip(self.adapters.values(), old):
                a.multiplier = value

    def close(self):
        for host, adapter in reversed(self.hosts):
            if host.forward == adapter.forward:
                host.forward = adapter.org_forward
            elif host.forward != adapter.org_forward:
                raise RuntimeError("Foreign forward mutation; refusing to remove it")
        self.hosts.clear()


class Native(ARLastHiddenResponseAdapter):
    family = "cohere2_native_text"

    def __init__(self, model, *, checkpointing=True):
        slider = Slider(model)
        try:
            super().__init__(model, slider, model_identity=MODEL_IDENTITY,
                checkpoint_modules=tuple(model.model.layers) if checkpointing else (),
                penalty_input_domain="continuous")
        except BaseException:
            slider.close()
            raise

    def _hidden(self, ids):
        tokens = torch.tensor([ids], dtype=torch.long, device=next(self.model.parameters()).device)
        result = self.model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
            use_cache=False, return_dict=True)
        return result.last_hidden_state[:, -1, :]

    def metadata(self):
        return {"protocol": PROTOCOL, "model_identity": MODEL_IDENTITY,
            "target_convention": "same gold answer prefix; private verified answer only in teacher context",
            "student_conditioning": "exact frozen evaluation prompt",
            "response_coordinates": "native final token hidden state, full width 2048",
            "adapter_names": list(self.network.names), "ema_decay": None}

    def close(self):
        self.close_checkpointing()
        self.slider.close()

    def response(self, *, text, strength=1., seed=0):
        from .evaluate import NativeGenerator, tokenizer
        with self.slider.scaled(strength):
            return NativeGenerator(self.model, tokenizer())([text])[0]


def initialize(native, critic, cohort):
    from particlegan.init import R2Normal
    from particle_sliders.initialization import ComponentInitialization, initialize_fresh_components
    # Tiny Aya ties its LM head to the token embedding. Declare that storage
    # exactly once: core correctly refuses overlapping parent/alias scopes.
    assert native.model.lm_head.weight is native.model.model.embed_tokens.weight
    scopes = {"pretrained_trunk_and_tied_head": ComponentInitialization(native.model.model, "keep")}
    for owner_name, owner in [("adapter", native.slider), ("critic", critic)]:
        for name, layer in owner.named_modules():
            if type(layer) is nn.Linear:
                full = f"{owner_name}.{name}"
                seeds = {n: int(digest({"seed": SEED, "name": full+"."+n})[:15], 16) for n,_ in layer.named_parameters()}
                scopes[full] = ComponentInitialization(layer, "deterministic_orthogonal", seeds)
    holder = nn.Module()
    holder.register_parameter("z", native.particles)
    scopes["bank"] = ComponentInitialization(holder, "deterministic_orthogonal", {"z": SEED+900},
        distributions={"z": R2Normal(mean=0., std=1.)})
    receipt = initialize_fresh_components(scopes, cohort_id=cohort,
        source_identity={"binding_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "core": "63f6e3942340d8aadca816800d19fb1ac88d92a9", "policy": "declared-host-RMS-QR-zero-up-R2-v1"},
        completed_steps=0, restored_from=None, optimizers=())
    assert all(torch.count_nonzero(a.lora_up.weight) == 0 for a in native.slider.adapters.values())
    return receipt


def build_cache(data: Path, out: Path, model, tok, resource):
    if out.exists():
        raise FileExistsError("Teacher cache is immutable; load its saved rows")
    rows = read_jsonl(data / "train.jsonl")
    if len(rows) != 64 or any(r["split"] != "train" for r in rows):
        raise ValueError("Only the frozen 64 training examples may supply targets")
    prepared, index = [], []
    # No slider is attached; the full pretrained model stays frozen.
    with torch.inference_mode():
        for example in rows:
            resource.check()
            neutral = tok.apply_chat_template(messages(example), add_generation_prompt=True)
            positive = tok.apply_chat_template(messages(example, teacher=True), add_generation_prompt=True)
            answer_ids = tok.encode(example["gold"], add_special_tokens=False)
            # All token prediction positions, including the first position,
            # use exactly the same gold continuation in student and teacher.
            for position in range(len(answer_ids)):
                history = tuple(answer_ids[:position])
                nr, pr = tuple(neutral)+history, tuple(positive)+history
                states = []
                for ids in (nr, pr):
                    tokens = torch.tensor([ids], device="cuda:0")
                    h = model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False, return_dict=True).last_hidden_state[:, -1, :]
                    states.append(h.float().detach().cpu().clone())
                prepared.append(PreparedAudioResponse(AudioRow(nr, pr, position, 1.), states[1], states[0]))
                index.append({"id": example["id"], "position": position, "history_ids": history,
                    "neutral_ids": nr, "teacher_ids": pr, "target_token": answer_ids[position]})
            print(json.dumps({"cached_case": example["id"], "response_rows": len(prepared)}), flush=True)
    out.mkdir(parents=True)
    atomic_save(out / "responses.pt", prepared)
    write_json(out / "index.json", {"protocol": PROTOCOL, "base": MODEL_IDENTITY,
        "train_sha256": digest(rows), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "teacher_information_evaluated": False, "cases": 64, "rows": index})


def validate(options):
    if options.get("ema_decay") is not None or options.get("base_precision") != "bfloat16":
        raise ValueError("RAW/no weight EMA and native BF16 required")
    if options["batch"] != 2 or options["steps"] != 128 or options["g_lr"] not in (.012, .004, .001):
        raise ValueError("Frozen three-rate/batch2/horizon128 cohort required")
    if not Path(options["cache"]).is_dir():
        raise ValueError("Saved training-only teacher cache required")


def prepare(options, *, restored_from=None):
    from .evaluate import load_base
    torch.manual_seed(SEED)
    model = load_base()
    native = Native(model)
    try:
        critic = AudioResponseCritic(2048, 2052).to(device="cuda:0", dtype=torch.float32)
        receipt_path = Path(options["initialization_receipt"])
        if restored_from is None:
            if receipt_path.exists():
                raise ValueError("Fresh initialization receipt exists; resume saved state instead")
            receipt = initialize(native, critic, options["cohort_id"])
            write_json(receipt_path, {"sha256": receipt.sha256, "receipt": receipt.as_dict()})
        else:
            if not receipt_path.is_file():
                raise ValueError("Resume requires the original initialization receipt")
        cache = Path(options["cache"])
        original = torch.load(cache / "responses.pt", map_location="cpu", weights_only=False)
        rows = [PreparedAudioResponse(p.row, p.target.to("cuda:0"), p.neutral.to("cuda:0")) for p in original]
        rate = options["g_lr"]
        game = ConditionalPG09(native, rows, critic, conditions=audio_conditions(rows), particles=native.particles,
            response_domain="continuous_vector", penalty_input_domain="continuous",
            batch_size=2, seed=SEED, total_steps=128, response_scale=1.,
            recipe_overrides={"lr": rate, "d_lr_mult": .018/rate, "prior_lr_mult": .03/rate})
        actual_g = [g["lr"] for g in game.opt_g.param_groups]
        assert sorted(actual_g) == sorted([rate, .03])
        assert all(abs(g["lr"]-.018) < 1e-12 for g in game.opt_d.param_groups)
        assert game.recipe.ema_decay == 0.0
        spec = INFERENCE.specification(native, model_identity=MODEL_IDENTITY, protocol=PROTOCOL,
            configuration={"rank": 8, "alpha": 8, "bank_shape": [128,4], "ema_decay": None})
        return Binding(game, native.close, inference=spec)
    except BaseException:
        native.close()
        raise


BACKEND = ResponseBackend(NAME, CAPS, prepare, validate)


def validate_inference(options, spec):
    if options or spec.configuration != {"rank":8,"alpha":8,"bank_shape":[128,4],"ema_decay":None}:
        raise ValueError("Native inference uses only the pinned base and exported adapter")


def prepare_inference(options, spec):
    from .evaluate import load_base
    native = Native(load_base(), checkpointing=False)
    return InferenceBinding(native.network, native_identity(native, model_identity=MODEL_IDENTITY, protocol=PROTOCOL),
        native.close, response=native.response)


INFERENCE = InferenceBackend(NAME, CAPS, prepare_inference, validate_inference)
