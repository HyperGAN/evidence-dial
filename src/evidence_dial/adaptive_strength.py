"""Activation-dependent steering on native Tiny Aya answer prediction positions."""

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from functools import partial
from math import isfinite
from typing import Literal

import torch
from particle_sliders.adaptive_steering import (
    ActivationGate,
    CLABSParameters,
    activation_projection,
    steer_residual,
)
from particle_sliders.prompt_steering import PromptSteering


@dataclass(frozen=True, kw_only=True)
class CachedGate:
    """An explicitly configured paper gate and intervention at a decoder input."""

    layer: int
    method: Literal["cfp", "clabs", "random"]
    parameter: float
    intervention: Literal["residual", "raw_adapter"]
    evidence_ref: str
    direction: torch.Tensor | None = None
    seed: int | None = None
    clabs: CLABSParameters | None = None
    preserve_norm: bool = False
    projection_bounds: tuple[float, float] | None = None
    include_answer_onset: bool = False


@dataclass(frozen=True, kw_only=True)
class CachedPromptSteering:
    """A fitted S-PSR module for the input residual of the named decoder layer."""

    layer: int
    fitted: PromptSteering
    evidence_ref: str
    include_answer_onset: bool = False


@dataclass(frozen=True, kw_only=True)
class AllDecoderPromptSteering:
    """Fitted A-PSR modules in native decoder input order, one per layer."""

    fitted: tuple[PromptSteering, ...]
    evidence_ref: str
    include_answer_onset: bool = False


@contextmanager
def adaptive_token_strengths(model, scale_adapter, *, prefix_tokens, max_tokens, strength, steering):
    """Scope the chosen interventions and collect the per-request replay trace."""
    if not isfinite(strength) or strength == 0:
        raise ValueError("Adaptive steering requires a finite, nonzero strength")
    if not isinstance(steering, (CachedGate, CachedPromptSteering, AllDecoderPromptSteering)):
        raise TypeError("Expected CachedGate, CachedPromptSteering or AllDecoderPromptSteering")
    if not isinstance(steering.evidence_ref, str) or not steering.evidence_ref.strip():
        raise ValueError("Adaptive steering requires a calibration or fit evidence reference")
    onset = steering.include_answer_onset
    if type(onset) is not bool:
        raise TypeError("include_answer_onset must be Boolean")
    layers = model.model.layers
    if model.config.model_type != "cohere2":
        raise ValueError("Adaptive steering requires a native Cohere2 decoder")
    all_decoder = isinstance(steering, AllDecoderPromptSteering)
    if all_decoder:
        indices = tuple(range(len(layers)))
        probes = steering.fitted
        if (not indices or not isinstance(probes, tuple) or len(probes) != len(indices)
                or len({id(probe) for probe in probes}) != len(probes)):
            raise ValueError("Supply a different fitted A-PSR module for each native decoder layer")
    else:
        if type(steering.layer) is not int or not 0 <= steering.layer < len(layers):
            raise ValueError("Select a native Cohere2 decoder input by zero-based layer index")
        indices = (steering.layer,)
        probes = (steering.fitted,) if isinstance(steering, CachedPromptSteering) else ()

    direction = None
    gate = None
    intervention = "residual"
    method = "a_psr" if all_decoder else "s_psr"
    if isinstance(steering, CachedGate):
        method = steering.method
        intervention = steering.intervention
        if intervention not in ("residual", "raw_adapter"):
            raise ValueError("The intervention must be residual or raw_adapter")
        if intervention == "raw_adapter" and (steering.layer != 0 or steering.preserve_norm
                                               or steering.projection_bounds is not None):
            raise ValueError("RAW adapter gating reads layer 0 before all adapter targets")
        if intervention == "raw_adapter" and onset:
            raise ValueError("Answer-onset steering requires a direct residual policy")
        device = next(layers[steering.layer].parameters()).device
        needs_direction = method != "random" or intervention == "residual"
        if needs_direction:
            direction = steering.direction
            if (not isinstance(direction, torch.Tensor) or not direction.is_floating_point()
                    or direction.shape != (model.config.hidden_size,)):
                raise ValueError("Supply the fitted direction at the selected Tiny Aya decoder input")
            direction = direction.detach().to(device=device, dtype=torch.float32)
            length = direction.norm()
            if not torch.isfinite(length) or length == 0:
                raise ValueError("The fitted direction must be finite and nonzero")
            direction = direction / length
        gate = ActivationGate(method, steering.parameter, seed=steering.seed, clabs=steering.clabs)
    else:
        for index, probe in zip(indices, probes, strict=True):
            device = next(layers[index].parameters()).device
            if (not isinstance(probe, PromptSteering) or probe.probe.in_features != model.config.hidden_size
                    or any(p.device != device for p in probe.parameters())):
                raise ValueError(f"Place the fitted PSR module for layer {index} on its native device")

    applied = []
    trace = []
    receipt = {"method": method, "intervention": intervention,
               "evidence_ref": steering.evidence_ref, "cached_steps": trace}
    if all_decoder:
        receipt["decoder_input_layers"] = list(indices)
    else:
        receipt["decoder_input_layer"] = steering.layer
    if gate is not None:
        receipt.update(parameter=steering.parameter, seed=steering.seed,
                       clabs=asdict(steering.clabs) if steering.clabs is not None else None,
                       preserve_norm=steering.preserve_norm, projection_bounds=steering.projection_bounds)
    position = None
    layer_records = []
    scale_scope = ExitStack()

    def before_forward(_module, _args, kwargs):
        nonlocal position
        if position is not None:
            raise RuntimeError("The selected decoder input was not reached in the previous forward")
        step = len(applied)
        ids, mask, cached = kwargs.get("input_ids"), kwargs.get("attention_mask"), kwargs.get("cache_position")
        width = 1 if step else prefix_tokens
        expected_last = prefix_tokens + step - 1
        if (step >= max_tokens or ids is None or ids.shape != (1, width)
                or mask is None or mask.shape != (1, expected_last + 1)
                or cached is None or cached.shape != (width,) or cached[-1].item() != expected_last):
            raise RuntimeError("Tiny Aya cached evaluation differs from the expected absolute position")
        position = step

    def before_decoder(index, _module, args):
        nonlocal position
        if position is None or index != indices[len(layer_records)]:
            raise RuntimeError("Tiny Aya decoder inputs did not run once each in the requested order")
        record = {"effective_strength": 0.0}
        updated = None
        if position or onset:
            hidden = args[0]
            width = 1 if position else prefix_tokens
            if hidden.shape != (1, width, model.config.hidden_size):
                raise RuntimeError("Unexpected native Tiny Aya decoder residual shape")
            active = hidden[:, -1:] if position == 0 else hidden
            edited = active
            if gate is None:
                probe = probes[len(layer_records)]
                edited, coefficient = probe(active, strength)
                alpha = torch.as_tensor(strength, dtype=coefficient.dtype, device=coefficient.device)
                effective = (alpha * coefficient).item()
                if not isfinite(effective):
                    raise ValueError("PSR produced a nonfinite effective strength")
                record["coefficient"] = coefficient.item()
            else:
                score = (activation_projection(active, direction, strength=strength).item()
                         if method != "random" else None)
                decision = gate.step(score)
                effective = strength if decision.fired else 0.0
                record.update(alignment=score, probability=decision.probability, fired=decision.fired)
                if intervention == "raw_adapter" and decision.fired:
                    scale_scope.enter_context(scale_adapter(strength))
                if intervention == "residual":
                    edited = steer_residual(active, direction, strength=strength, gate=decision.fired,
                                            preserve_norm=steering.preserve_norm,
                                            projection_bounds=steering.projection_bounds)
            record["effective_strength"] = effective
            if edited is not active:
                updated = edited if position else torch.cat((hidden[:, :-1], edited), dim=1)
        layer_records.append(record)
        if len(layer_records) == len(indices):
            step = {"output_index": position, "cache_position": prefix_tokens + position - 1}
            if all_decoder:
                step["layers"] = list(layer_records)
                applied.append([record["effective_strength"] for record in layer_records])
            else:
                step.update(record)
                applied.append(record["effective_strength"])
            if position:
                trace.append(step)
            elif onset:
                receipt["answer_onset_step"] = step
            layer_records.clear()
            position = None
        if updated is not None:
            return (updated, *args[1:])
        return None

    def after_forward(_module, _args, _output):
        scale_scope.close()

    with ExitStack() as handles:
        handles.callback(model.register_forward_pre_hook(before_forward, with_kwargs=True).remove)
        for index in indices:
            handle = layers[index].register_forward_pre_hook(partial(before_decoder, index))
            handles.callback(handle.remove)
        handles.callback(model.register_forward_hook(after_forward, always_call=True).remove)
        handles.callback(scale_scope.close)
        yield applied, receipt
        if gate is not None:
            receipt["usage"] = asdict(gate.usage)
