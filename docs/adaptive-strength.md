# Activation-dependent Tiny Aya steering

`evidence_dial.portable.Portable.generate` accepts an opt-in `adaptive` policy
and an explicit signed `strength`. The default scope starts with unedited
prefill and steers the subsequent cached forwards through native EOS. A direct
residual policy can set `include_answer_onset=True` to also steer the final
prompt token, which predicts output index `0`. Earlier prompt tokens retain
their original residuals. Output index `1` remains the first cached prediction.
The existing RAW gate, constant strength and `step_gains` retain their scope.

## Choose the intervention

| Policy | Activation observed | Edit |
| --- | --- | --- |
| `CachedGate(intervention="residual")` | Input to the calibrated Cohere2 decoder layer | Apply the shared CFP or CLABS gate with the paper's fixed-direction residual edit. Use `method="random"` for the same-direction budget control. |
| `CachedGate(intervention="raw_adapter")` | Layer 0 input, before every RAW attention target | Gate the saved routed adapter with CFP or CLABS. This uses the paper's allocation equation with the learned RAW adapter as the edit. Layer 0 reads a token embedding before contextualization. |
| `CachedPromptSteering` | Input to the decoder layer where the S-PSR module was fitted | Apply the shared fitted affine-ReLU coefficient and rank-one residual edit. |
| `AllDecoderPromptSteering` | Input to every native decoder layer in order | Apply one distinct fitted shared `PromptSteering` module per layer for A-PSR. |

Direct residual paths run with the RAW adapter off. Replaying the RAW
fixed-strength control for a cached-only comparison uses a prefill gain of
zero. The previously archived RAW256 timing result has its own prefill gain
and stays reproducible through `step_gains`.

```python
from evidence_dial.adaptive_strength import AllDecoderPromptSteering, CachedGate, CachedPromptSteering
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
# For A-PSR, load one distinct fitted module per decoder layer in native order.
a_psr_policy = AllDecoderPromptSteering(
    fitted=tuple(fitted_modules), evidence_ref=fit_manifest, include_answer_onset=True,
)
with Portable(raw256_directory) as engine:
    result = engine.generate(context, question, strength=calibrated_strength, adaptive=policy)
```

Use the recorded absolute cache positions, measured alignment or PSR
coefficient, effective strengths, and gate counts in `adaptive_steering` to
replay and audit the request. The trace includes the decoder input layer,
base revision, evidence reference and, for RAW gating, the adapter checksum.
The onset record appears in `answer_onset_step` when enabled. The
`cached_steps` field retains the records after prefill. A-PSR returns
`effective_layer_strengths` indexed by output position, then decoder layer.
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

## Native A-PSR research pilots

Two separately trained fits used aligned original Tiny Aya activations from
42 qualified positive prompt pairs. Each fit trained 36 decoder-input PSR
modules jointly for 630 updates. The matched constant control used each
layer's mean fitted coefficient from live 128-case development rollouts.
The cached-only fit and the answer-onset fit used their own aligned position
scopes and distinct held-out panels.

| Native held-out pilot | OFF | Matched constant | A-PSR |
| --- | ---: | ---: | ---: |
| Cached-only, Policies | 160/256 | 160/256 | 160/256 |
| Answer onset, balanced grounding confirmation | 353/768 | 349/768 | 339/768 |

The cached-only scores tied. The answer-onset A-PSR fit scored 14 fewer
correct than OFF and 10 fewer than its matched constant control. The qualified
independent observer passed all 768 saved replies in each answer-onset arm.
It reused verdicts for byte-identical reply text across cases. Integrity
remains unmeasured for the cached-only pilot. Both RAW fits remain opt-in
research. [Pilot evidence](../artifacts/evidence/tiny-aya-a-psr-pilots.json)
