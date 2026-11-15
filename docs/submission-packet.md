# Particle Sliders for Tiny Aya — hackathon submission packet

Team: **Particle Sliders**. **Track 1, section 7: Preference and style alignment.**
The owner completes the submission form; this repository claims no form-submission receipt.

| Submission field | Value |
| --- | --- |
| Main source and thorough README | https://github.com/HyperGAN/tiny-aya-particle-sliders |
| Complete recorded-output demo | https://huggingface.co/spaces/ntc-ai/tiny-aya-particle-sliders |
| Winning raw behavior adapter | https://huggingface.co/ntc-ai/tiny-aya-particle-sliders |
| Full data, checkpoints, predictions and audits | https://huggingface.co/datasets/ntc-ai/tiny-aya-particle-sliders |
| Public model collection | https://huggingface.co/collections/ntc-ai/evidence-dial-models-and-adapters-6af28e2624ee9d502264a99f |
| Public dataset collection | https://huggingface.co/collections/ntc-ai/evidence-dial-datasets-and-benchmark-6af2746d22e8aefe32cafb87 |

Winner: **step 1,024, strength 1, G 0.0000625**. [Direct native weights](https://huggingface.co/ntc-ai/tiny-aya-particle-sliders/resolve/b7e7f256035b9eb8d4ded4d01af52effff2e51fc/adapter-native.pt?download=true) · [Winner record](https://huggingface.co/ntc-ai/tiny-aya-particle-sliders/blob/main/winner.json) · [Method and math](method.md).

## Abstract

Particle Sliders for Tiny Aya turns written response preferences into an adjustable adapter for frozen Tiny Aya Global 3.35B. Its target is sympathetic, constructive support with independent judgment. For each of 64 TRAIN questions, the unmodified model generates complete responses under three unrestricted behavioral instructions and a neutral control. NanoGPT Jev filters successful contexts; one best qualifying context per question yields 37 paired first-answer hidden states. Neutral and positive inputs retain the same question and empty continuation history, while only positive receives the behavioral instruction.

The unchanged ParticleGAN 0.9.5 formulation trains a routed adapter, critic and particle bank while the backbone stays frozen. Two fresh learning-rate cohorts complete 1,024 updates each, with raw weights and no weight EMA. Selection uses finished intended effect ON minus matched OFF across all 24 declared DEV settings. At inference, the adapter retains the ordinary question, adds no behavioral prompt, and exposes an intensity control.

On 64 fresh synthetic questions, the preselected adapter scores 0.7327, versus 0.6642 OFF and 0.6976 prompt-only. Its gains are +0.0685 and +0.0351; both individual paired bootstrap intervals exclude zero. All 384 comparison answers reach native EOS. Sympathy judgments rise, while the separate sycophancy diagnostic does not fall versus OFF. Earlier negative results remain public. One training seed, related synthetic scenario families, a shared curation/evaluation judge and first-state supervision limit the conclusions.

## Results and value

| Frozen condition, 64 fresh questions each | Mean behavior score | Change from OFF |
| --- | ---: | ---: |
| OFF — unmodified base | 0.664180 | +0.000000 |
| Unrestricted prompt-only | 0.697578 | +0.033398 |
| PG strength 1 — selected on DEV | 0.732656 | +0.068477 |
| PG strength 0.25 — descriptive | 0.695195 | +0.031016 |
| PG strength 0.5 — descriptive | 0.703555 | +0.039375 |
| Earlier expanded-study adapter — fixed reference | 0.668086 | +0.003906 |

The added value is the concrete connection from written preferences to curated
internal-state targets, original ParticleGAN training, an adjustable raw adapter
and auditable finished outputs. The selected model keeps the ordinary user
question and receives no extra behavioral instruction. The gallery contains all
64 fresh questions, both controls, every frozen strength and the earlier reference.

The primary gain over OFF is +0.068477, individual paired 95% interval
[+0.051211, +0.085392]; over prompt-only it is +0.035078
[+0.013358, +0.058320]. Sympathy rises, but sycophancy probability does not
fall versus OFF. This is one synthetic corpus/seed and a shared model judge,
with no human preference or matched behavior-SFT study. Earlier negative results
remain public. Separate QA tied SFT on DEV, while held-out results favored SFT.

## Attribution and delivery

CuratorKIT handles schema/deduplication; AuditKIT wraps native generation and Jev.
AlignTune supplies the separate QA SFT baselines. ParticleGAN, Particle Sliders
and Meta Glimmer are pre-existing foundations; the entry is the Tiny Aya
integration, curation/training/evaluation pipeline and recorded-output explorer.

Both 1,024-update runs, all 24 DEV cells and all 384 fresh answers are verified.
Weights are raw, with model-weight EMA disabled. The README supplies method,
full tables, inference/source entry points, improvement/regression examples and
honest limitations. Tiny Aya-derived weights/states use CC-BY-NC-4.0. The Space
uses saved outputs; fresh-machine replication and browser visual/click QA are
not claimed without separate evidence. No paper or owner form submission is claimed.
