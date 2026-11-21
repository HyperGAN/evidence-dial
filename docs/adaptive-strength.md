# Activation-dependent Tiny Aya steering

`evidence_dial.portable.Portable.generate` accepts an opt-in `adaptive` policy
and an explicit signed `strength`. Each request starts with unedited prefill.
The policy observes one pre-intervention residual per subsequent cached forward
and records one decision through native EOS. Output index `0` is predicted by
prefill. Output index `1` is the first cached prediction. The API retains
constant strength and `step_gains` for the existing RAW controls.

## Choose the intervention

| Policy | Activation observed | Edit |
| --- | --- | --- |
| `CachedGate(intervention="residual")` | Input to the calibrated Cohere2 decoder layer | Apply the shared CFP or CLABS gate with the paper's fixed-direction residual edit. Use `method="random"` for the same-direction budget control. |
| `CachedGate(intervention="raw_adapter")` | Layer 0 input, before every RAW attention target | Gate the saved routed adapter with CFP or CLABS. This uses the paper's allocation equation with the learned RAW adapter as the edit. Layer 0 reads a token embedding before contextualization. |
| `CachedPromptSteering` | Input to the decoder layer where the S-PSR module was fitted | Apply the shared fitted affine-ReLU coefficient and rank-one residual edit. |

Both direct residual paths run with the RAW adapter off. Replaying the RAW
fixed-strength control for a cached-only comparison uses a prefill gain of
zero. The previously archived RAW256 timing result has its own prefill gain
and stays reproducible through `step_gains`.

```python
from evidence_dial.adaptive_strength import CachedGate, CachedPromptSteering
from evidence_dial.portable import Portable

# Load direction and all gate settings from the selected calibration record.
policy = CachedGate(
    layer=calibrated_layer,
    method=calibrated_method,
    parameter=calibrated_threshold_or_bias,
    intervention="residual",
    direction=fitted_direction,
    seed=gate_seed,
    clabs=calibrated_clabs_parameters,
    preserve_norm=preserve_norm,
    projection_bounds=projection_bounds,
    evidence_ref=calibration_manifest,
)
# For an S-PSR run, use a fitted shared PromptSteering on the same device as
# the selected decoder layer and the recorded positive/neutral pair manifest.
psr_policy = CachedPromptSteering(
    layer=fitted_layer, fitted=fitted_psr_module, evidence_ref=fit_manifest,
)
with Portable(raw256_directory) as engine:
    result = engine.generate(context, question, strength=calibrated_strength, adaptive=policy)
```

Use the recorded absolute cache positions, measured alignment or PSR
coefficient, effective strengths, and gate counts in `adaptive_steering` to
replay and audit the request. The trace includes the decoder input layer,
base revision, evidence reference and, for RAW gating, the adapter checksum.
The calibration or fit manifest must identify the decoder input boundary,
checkpoint, activation dtype, artifacts and dataset versions. Use the
[shared equations and calibration APIs](https://github.com/HyperGAN/particle-sliders/blob/b1f780decc516808cfd3019d5e5b674457161f86/docs/strength-schedules.md#activation-dependent-steering).

## Qualify against the frozen grounding benchmark

- For CFP/CLABS, fit the direction from contrasting training examples using an
  SAE from the exact pinned Tiny Aya decoder input. Calibrate the CFP threshold
  or CLABS bias through active native rollouts on the 128-case development
  split, using the same matched cases for every candidate. Freeze the direction,
  controller values, requested budget and edit configuration before evaluation.
- For S-PSR, build prompt pairs from the 64 training cases. Generate and qualify
  a response from the positive prompt, then replay the same response tokens on
  both sides. Collect aligned states at and after the chosen decoder input.
  Train the shared PSR module against positive-prompt activations while the
  neutral prompt follows the edited forward. Select inference strength on the
  development split. Existing final-hidden teacher artifacts alone cannot
  supply the aligned per-layer activations for this fit.
- Use the family-disjoint 256-case Policies holdout for the finished-output
  comparison. Use native OFF, matched strength and allocation controls, and
  record realized gate usage. Score equal-case exact grounded-answer success
  ON minus matched OFF and run the separate output-integrity check. The 128-case
  [archived RAW256 timing result](../artifacts/evidence/grounding-raw256-strength-schedule.json)
  is a development result. [Frozen benchmark and splits](reproduce.md)

The cached paths have been verified with a small native Cohere2 fixture. A
Tiny Aya calibration artifact, fitted S-PSR probe and adaptive finished-output
gain still require qualification. The packaged path needs caller-supplied
settings, so it installs no adaptive preset.
