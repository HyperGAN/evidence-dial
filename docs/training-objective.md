# The original answer-hinted particle objective

This document describes the original64-case answer-hinted target recipe. The
[corrected Jev-curated workflow](method-repair.md) retains the shared architecture
and game below while changing target curation and provenance.

The adapter uses the shared Particle Sliders activation-conditioned routed
projection. Tiny Aya stays frozen. At every q/k/v/o projection in36 layers,
let h be the current native activation, x=down(h), and P the learned128×4 bank:

```text
q = router(x)
z = softmax(q @ P.T / sqrt(4)) @ P
delta = up(MLP(concat(x, z)))
output = frozen_projection(h) + strength * delta
```

The rank is8. The same learned bank is shared across projections; routing
weights depend on each current activation. No independently sampled particle
index or random z is supplied to the language model. This is the actual shared
`TinyParticleProjection` / `RoutedMLP`, rather than a new prompt-level adapter.
All2593 learned tensors, including the bank, passed connected-gradient checks.
Strength0 bypasses the adapter exactly. Training and inference use the same
routing. RAW weights are retained; model-weight EMA is disabled.

For each of64 training examples, neutral and teacher are paired by the exact
same documents, question and preceding gold-answer tokens. Both are frozen
base-model forward passes in eval mode. Neutral uses the exact evaluation
prompt. Teacher adds only:

```text
Verified answer (private teacher information): <gold answer>
```

Before each next gold token, capture the native final2048-dimensional hidden
state. There are146 cached positions. The answer prefix is teacher-forced and
identical across the two passes; it is not two independently sampled answers.
The extra hint changes the teacher's internal activations and sequence length.
We do not freeze one intermediate activation and replay it into both branches.
Neither cache branch has an attached particle adapter. The adapted student is
subsequently evaluated on the neutral input/prefix and trained toward the
cached teacher hidden response. Only the training split supplies these targets.
The completed cache is reused, never regenerated to improve a result.

The official conditional ParticleGAN game operates on this response residual.
Within each generator or discriminator phase, it samples row IDs and one noise
tensor epsilon, then constructs:

```text
real = epsilon
fake = epsilon + (student_hidden - teacher_hidden) / response_scale
```

Thus real and fake share the same epsilon and corresponding sampled rows within
a phase. Generator and discriminator phases have separate row/noise streams.
This critic noise is not an internal Tiny Aya latent. The critic is conditioned
on the full frozen neutral hidden state plus input lengths, answer position and
strength; it does not compare unrelated unconditioned answers. The official
optimizer/penalty/scalar-statistic formulas are preserved.

At inference, only the documents, question, shared strong prompt, base and
learned adapter are used. No verified-answer hint, gold prefix, teacher, critic,
optimizer or training cache is supplied. Native greedy decoding produces the
finished answer. The current objective is adversarial hidden-response matching,
not direct answer-logit or token cross-entropy optimization. AlignTune LoRA is
the separate gold-completion cross-entropy comparison.

The adapter formulation and objective are different claims: the routed bank
uses internal activations, while the chosen text training response is the final
hidden state. Lower teacher-response SSE or adversarial loss cannot establish
better answers. Development selection and all reported effects use only matched
finished strict success ON minus OFF. The saved model was selected before any
holdout output. Stronger tests do not change that selection or launch training.

Sources: `src/evidence_dial/binding.py`, `protocol.py`, and the pinned core's
`routed_projection.py`, `reference.py:RoutedMLP`, `pg09.py:ConditionalPG09`, and
`pg09_audio.py:audio_conditions`. Exact source revisions and historical content
hashes are retained in the reproduction bundle and training archive.
