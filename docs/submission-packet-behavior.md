# Evidence Dial — hackathon submission packet

Team: **Particle Sliders**. Category: **Track 1, section 7 — Preference and style alignment**.
The owner fills the submission form; this repository does not claim a form submission receipt.

| Submission field | Value |
| --- | --- |
| Main project and thorough README | https://github.com/HyperGAN/evidence-dial |
| Fresh complete-answer confirmation and archived earlier studies | https://huggingface.co/spaces/ntc-ai/evidence-dial |
| Behavior model and raw adapter | https://huggingface.co/ntc-ai/evidence-dial-behavior-particle |
| Data, all predictions, training logs and audits | https://huggingface.co/datasets/ntc-ai/evidence-dial-behavior |
| Model collection | https://huggingface.co/collections/ntc-ai/evidence-dial-models-and-adapters-6af28e2624ee9d502264a99f |
| Dataset collection | https://huggingface.co/collections/ntc-ai/evidence-dial-datasets-and-benchmark-6af2746d22e8aefe32cafb87 |

## Abstract

Evidence Dial connects language-defined response preferences to a learned
behavior control for frozen Tiny Aya Global 3.35B. For each of 64 training
questions, the base model explores 16 behavioral instructions. NanoGPT Jev
judges sympathetic, constructive support with independent judgment. The expanded
dataset retains all 138 passing contexts across 37 questions. CuratorKIT
validates and deduplicates the corpus; AuditKIT connects native generation to a
fixed evaluator.

Positive and neutral contexts retain the same user question and empty
continuation history. Only the positive context receives the behavioral system
instruction. We capture frozen final hidden states before the first answer token
and train a routed particle adapter using the unchanged ParticleGAN formulation.
There is no SFT warm start, text-imitation loss, or weight EMA. At inference,
the adapter sees the ordinary question and its strength is adjustable.

Both fresh learning-rate cohorts finished 1,024 updates. All 24 development
settings were audited using complete native-EOS answers. Development selection
chose step 256, strength 1, scoring 0.6950 versus 0.6620 OFF. Longer training
did not improve that maximum. On a fresh 32-question confirmation, the locked
adapter scored 0.6739 versus 0.6867 OFF and 0.6494 prompt-only. Its change from
OFF was −0.0128, with an individual paired 95% bootstrap interval [−0.0473,
+0.0208]. The expanded study does not establish an improvement over the base
model; stronger steering regresses.

The contribution is the reusable curation, internal-state training, evaluation
and recorded-output pipeline, including its failures. A separate QA study tied
SFT on DEV but favored SFT on held-out data. One synthetic corpus, one training
seed, a shared curation/evaluation judge and first-state supervision limit the
claims.

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

The main README supplies the abstract, complete results, all 24 DEV settings, native-EOS examples, method, artifact links and honest limitations. Both expanded 138-context cohorts completed 1,024 updates; all 160 fresh-confirmation answers are independently verified. The locked selected adapter scored 0.673906 versus OFF 0.686719 and prompt-only 0.649375, with uncertain differences. There is no demonstrated improvement over the base model, no greater-caring claim, and no SFT superiority claim. Earlier 37-pair/64-token and fixed-checkpoint studies remain archived. No paper or owner-form submission is claimed. Tiny Aya-derived weights/states retain CC-BY-NC-4.0 and upstream code licenses.

The Space is a recorded-output explorer. Live files and content hashes are verified; interactive browser QA requires an available browser host and is not claimed here.
