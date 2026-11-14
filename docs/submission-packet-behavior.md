# Evidence Dial — hackathon submission packet

Team: **Particle Sliders**. Category: **Track 1, section 7 — Preference and style alignment**.
The owner fills the submission form; this repository does not claim a form submission receipt.

| Submission field | Value |
| --- | --- |
| Main project and thorough README | https://github.com/HyperGAN/evidence-dial |
| Recorded held-out behavior demo | https://huggingface.co/spaces/ntc-ai/evidence-dial |
| Behavior model and raw adapter | https://huggingface.co/ntc-ai/evidence-dial-behavior-particle |
| Data, all predictions, training logs and audits | https://huggingface.co/datasets/ntc-ai/evidence-dial-behavior |
| Model collection | https://huggingface.co/collections/ntc-ai/evidence-dial-models-and-adapters-6af28e2624ee9d502264a99f |
| Dataset collection | https://huggingface.co/collections/ntc-ai/evidence-dial-datasets-and-benchmark-6af2746d22e8aefe32cafb87 |

## Abstract

We investigate preference and style alignment through internal-state training
with ParticleGAN. Frozen Tiny Aya explores 16 behavioral instructions for each
of 64 training questions. NanoGPT's Jev evaluates the replies against a fixed
rubric for sympathetic, constructive support with independent judgment.
Thirty-seven questions pass the declared threshold. CuratorKIT provides schema
validation and exact deduplication; AuditKIT connects native generation to the
frozen evaluator.

For each retained question, we capture the model's final hidden state before
its first answer token under a selected instruction. The neutral context is
the same user question without an added behavioral system message. Both
contexts have empty continuation history. A routed particle adapter learns
from these paired states using the unchanged ParticleGAN formulation, with a
frozen backbone, raw weights, and no cross-entropy text-imitation loss or
weight EMA.

Two learning rates run for 128 updates each. Checkpoint and strength selection
uses 16 development questions, retaining all maximum ties before a separate
32-question confirmation. The preselected adapter scores 0.5996 versus 0.5888 OFF on 32 held-out questions,
a +0.0108 change with an individual paired 95% bootstrap interval of
[−0.0122, +0.0347]. Prompt-only scores 0.6123. Strength 2 regresses to 0.5202. The saved-output demo exposes the identical
question across strengths and retains regressions.

Earlier QA testing reached the same development score as SFT, 174/192, but
held-out results favored SFT. Our contribution is a reusable connection
between language-defined preferences, curated internal states, and a learned
behavior control. The small synthetic corpus, one training seed, shared
curation/evaluation judge, and 64-token evaluation limit constrain the claims.

## Where we add value

A concrete connection from a written response preference to curated internal-state pairs, ParticleGAN training, and an adjustable raw-weight adapter. The same ordinary user question is retained at inference. The public explorer exposes all 32 held-out cases, the prompt-only baseline, and regressions. The reusable pipeline and complete records are the contribution; this small study does not demonstrate a quality advantage over prompting or SFT.

## Repositories and attribution

- [Evidence Dial](https://github.com/HyperGAN/evidence-dial): this entry's integration, corpus, curation, training/evaluation harness, audits, and demo.
- [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT): schema validation and exact deduplication.
- [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT): native callable evaluation with the fixed Jev scorer.
- [AlignTune](https://github.com/Lexsi-Labs/aligntune): SFT baselines in the separate QA study; it does not train the behavior PG adapter.
- [ParticleGAN](https://github.com/HyperGAN/ParticleGAN) and [Particle Sliders](https://github.com/HyperGAN/particle-sliders): pre-existing training and adapter libraries.
- [Meta Glimmer](https://github.com/HyperGAN/meta-glimmer-particle-slider/tree/813ea28dd37c156a23470d1ef9a4f1e418c581ac): paired prompts and frozen first-state capture reference.

## Delivery scope and limits

The main README supplies the abstract, complete development grid, held-out results table, actual examples, method, environment, artifact links, and honest limitations. Both original training roots and all five held-out arms completed and passed independent audit. The 138-context data derivative is prepared but untrained. No paper or claim of SFT superiority is part of this entry. Tiny Aya and derived weights/states are CC-BY-NC-4.0; retain upstream code licenses.

The Space is a recorded-output explorer. Live files and content hashes are verified; interactive browser QA requires an available browser host and is not claimed here.
