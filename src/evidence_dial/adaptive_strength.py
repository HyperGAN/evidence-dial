"""Activation-dependent steering on native Tiny Aya cached token evaluations."""

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True, kw_only=True)
class CachedPromptSteering:
    """A fitted S-PSR module for the input residual of the named decoder layer."""

    layer: int
    fitted: PromptSteering
    evidence_ref: str


@contextmanager
def adaptive_token_strengths(model, scale_adapter, *, prefix_tokens, max_tokens, strength, steering):
    """Edit only cached evaluations and collect the exact per-request replay trace."""
    if not isfinite(strength) or strength == 0:
        raise ValueError("Adaptive steering requires a finite, nonzero strength")
    if not isinstance(steering, (CachedGate, CachedPromptSteering)):
        raise TypeError("Expected CachedGate or CachedPromptSteering")
    if not isinstance(steering.evidence_ref, str) or not steering.evidence_ref.strip():
        raise ValueError("Adaptive steering requires a calibration or fit evidence reference")
    layers = model.model.layers
    if (model.config.model_type != "cohere2" or type(steering.layer) is not int
            or not 0 <= steering.layer < len(layers)):
        raise ValueError("Select a native Cohere2 decoder input by zero-based layer index")
    layer = layers[steering.layer]
    device = next(layer.parameters()).device
    direction = None
    gate = None
    intervention = "residual"
    method = "s_psr"
    if isinstance(steering, CachedGate):
        method = steering.method
        intervention = steering.intervention
        if intervention not in ("residual", "raw_adapter"):
            raise ValueError("The intervention must be residual or raw_adapter")
        if intervention == "raw_adapter" and (steering.layer != 0 or steering.preserve_norm
                                               or steering.projection_bounds is not None):
            raise ValueError("RAW adapter gating reads layer 0 before all adapter targets")
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
    elif (not isinstance(steering.fitted, PromptSteering)
          or steering.fitted.probe.in_features != model.config.hidden_size
          or any(p.device != device for p in steering.fitted.parameters())):
        raise ValueError("Place the fitted S-PSR module for this layer on its native device")

    applied = []
    trace = []
    receipt = {"method": method, "intervention": intervention, "decoder_input_layer": steering.layer,
               "evidence_ref": steering.evidence_ref, "cached_steps": trace}
    if gate is not None:
        receipt.update(parameter=steering.parameter, seed=steering.seed,
                       clabs=asdict(steering.clabs) if steering.clabs is not None else None,
                       preserve_norm=steering.preserve_norm, projection_bounds=steering.projection_bounds)
    position = None
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

    def before_decoder(_module, args):
        nonlocal position
        if position is None:
            raise RuntimeError("Tiny Aya evaluated the selected decoder input more than once")
        if position == 0:
            applied.append(0.0)
            position = None
            return None
        hidden = args[0]
        if hidden.shape != (1, 1, model.config.hidden_size):
            raise RuntimeError("Expected one native residual per cached Tiny Aya evaluation")
        records = {"output_index": position, "cache_position": prefix_tokens + position - 1}
        edited = hidden
        if gate is None:
            edited, coefficient = steering.fitted(hidden, strength)
            alpha = torch.as_tensor(strength, dtype=coefficient.dtype, device=coefficient.device)
            effective = (alpha * coefficient).item()
            if not isfinite(effective):
                raise ValueError("S-PSR produced a nonfinite effective strength")
            records["coefficient"] = coefficient.item()
        else:
            score = (activation_projection(hidden, direction, strength=strength).item()
                     if method != "random" else None)
            decision = gate.step(score)
            effective = strength if decision.fired else 0.0
            records.update(alignment=score, probability=decision.probability, fired=decision.fired)
            if intervention == "raw_adapter" and decision.fired:
                scale_scope.enter_context(scale_adapter(strength))
            if intervention == "residual":
                edited = steer_residual(hidden, direction, strength=strength, gate=decision.fired,
                                        preserve_norm=steering.preserve_norm,
                                        projection_bounds=steering.projection_bounds)
        records["effective_strength"] = effective
        trace.append(records)
        applied.append(effective)
        position = None
        if edited is not hidden:
            return (edited, *args[1:])
        return None

    def after_forward(_module, _args, _output):
        scale_scope.close()

    with ExitStack() as handles:
        handles.callback(model.register_forward_pre_hook(before_forward, with_kwargs=True).remove)
        handles.callback(layer.register_forward_pre_hook(before_decoder).remove)
        handles.callback(model.register_forward_hook(after_forward, always_call=True).remove)
        handles.callback(scale_scope.close)
        yield applied, receipt
        if gate is not None:
            receipt["usage"] = asdict(gate.usage)
