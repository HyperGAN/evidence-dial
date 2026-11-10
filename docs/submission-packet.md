# Evidence Dial submission packet

Owner handoff: Mikkel will fill in the final hackathon form. This file supplies
copyable project material; no submission receipt is claimed.

Team: **Particle Sliders**. Project: **Evidence Dial**.
Track **1**, problem statement **3**: grounded answers on an organisation's
documents, with explicit missing-evidence and conflict answers.

| Field | Value |
| --- | --- |
| Code and README article | https://github.com/HyperGAN/evidence-dial |
| Model collection | https://huggingface.co/collections/ntc-ai/evidence-dial-models-and-adapters-6af28e2624ee9d502264a99f |
| Dataset and benchmark collection | https://huggingface.co/collections/ntc-ai/evidence-dial-datasets-and-benchmark-6af2746d22e8aefe32cafb87 |
| AlignTune model | https://huggingface.co/ntc-ai/evidence-dial-aligntune |
| Policy corpus | https://huggingface.co/datasets/ntc-ai/evidence-dial-policies |
| Checkpoints and training logs | https://huggingface.co/datasets/ntc-ai/evidence-dial-training-archive |
| Particle adapter | Pending final guard, development selection and publication |
| Public demo | Pending final benchmark and static Space deployment |
| Benchmark results | Pending selection and complete matched holdout generation |

## Abstract draft — replace pending result before submission

Evidence Dial tests a learned grounding control for Cohere Labs' Tiny Aya
Global3.35B. Given the same question and supplied documents, the desired behavior
is to answer from evidence, return UNKNOWN when evidence is missing, and return
CONFLICT when documents disagree.

Our pipeline chains CuratorKIT, Particle Sliders, AlignTune and AuditKIT.
CuratorKIT validates and exports64 structured fictional policy examples.
Particle Sliders trains a routed rank8 adapter with the official ParticleGAN
game; AlignTune trains the required supervised LoRA on exactly the same cases.
AuditKIT evaluates native generations under a frozen strict answer protocol.
Every arm receives identical prompts, greedy decoding and a64-token output cap.
Weight EMA is disabled; checkpoints and exports retain current RAW values.

The primary benchmark contains256 cases from64 held-out policy families, with
equal answerable, missing, conflicting and altered-fact cases. External transfer
uses a frozen900-case FaithEval subset. Rate and strength selection uses only
128 development cases; family-clustered paired intervals quantify uncertainty.
All regressions, catastrophic stops and uncertain checks remain inspectable.

[Final primary and transfer effects will be inserted from the frozen benchmark.]

The public release includes adapter and dataset collections, exact commands,
training states and a saved-output HF explorer comparing OFF, particle strengths
1/2 and LoRA. Live ZeroGPU hosting hit the account's free Space limit, so the
plan's static fallback is used. Tiny Aya and derived weights are non-commercial
CC-BY-NC-4.0. The dataset, Cohere2 binding and evaluation pipeline were built for
this hackathon; Particle Sliders core is a pre-existing library.

## Account fields

The owner has supplied the registration email and confirmed Discord membership.
Personal form fields are intentionally kept out of this public packet. The
registration's Hugging Face username matching `ntc-ai` remains an owner credit
check. No further form action is requested from an agent.
