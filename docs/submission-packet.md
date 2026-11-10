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
| Original 64-case AlignTune model | https://huggingface.co/ntc-ai/evidence-dial-aligntune |
| Policy corpus | https://huggingface.co/datasets/ntc-ai/evidence-dial-policies |
| Checkpoints and training logs | https://huggingface.co/datasets/ntc-ai/evidence-dial-training-archive |
| Original answer-hinted particle comparison | https://huggingface.co/ntc-ai/evidence-dial-particle-hinted |
| Corrected Jev-curated particle | https://huggingface.co/ntc-ai/evidence-dial-particle |
| Same-48-case AlignTune control | https://huggingface.co/ntc-ai/evidence-dial-aligntune-curated |
| Successful samples, seeds and paired states | https://huggingface.co/datasets/ntc-ai/evidence-dial-curated-successes |
| Public demo | https://huggingface.co/spaces/ntc-ai/evidence-dial |
| Complete original and corrected benchmark | https://huggingface.co/datasets/ntc-ai/evidence-dial-benchmark |
| Corrected benchmark results | https://huggingface.co/datasets/ntc-ai/evidence-dial-benchmark/tree/bbebf36c42294ea0814a397147b5809f1e846b04/curated-retry-v1 |

## Abstract

Evidence Dial addresses Track 1, problem statement 3: answer from an organization's documents, return UNKNOWN when evidence is absent, and CONFLICT when equally authoritative records disagree.

Our CuratorKIT → Particle Sliders / AlignTune → AuditKIT pipeline uses Tiny Aya Global 3.35B. Frozen Tiny Aya generates explanations without reference-answer hints; strict checks and Jev 1.14 retain successful examples within a fixed seeded budget. 48 of 64 fictional policy cases qualify. Tokens, RNG states and hidden trajectories are preserved. Approved explanations supply positive context for 118 cached neutral/positive internal state pairs. A routed particle adapter and supervised LoRA both train for 128 updates, batch two, on the same source cases. Weight EMA is disabled.

Results depend strongly on the benchmark. On the repeated 900-case FaithEval subset, the development-selected particle scores 501/900 (55.7%), versus matched LoRA 270/900 (30.0%) and original 64-case LoRA 375/900 (41.7%). The paired gain over matched LoRA is 25.7 percentage points, 95% interval [22.7, 28.7]. A post hoc check accepting a correct choice label plus its exact option text reduces the gain to 19.9 points [17.2, 22.7]. Choice accuracy is then nearly tied; the substantial advantage is refusing unsupported answers.

Fresh balanced policy confirmation exposes the tradeoff: particle 209/768 (27.2%), matched LoRA 477/768 (62.1%), OFF 354/768 (46.1%). Particle minus LoRA is −34.9 points [−38.4, −31.4], driven by excess abstention. Thus the original perfect template score was not a ceiling, and the transfer win does not establish reliable organizational grounding.

The public release preserves all regressions, original comparisons, checkpoints, predictions, exact commands and a static HF explorer. Selection precedes new test outputs; intervals resample families or source questions. One training seed and absent verified FaithEval answerable QA controls limit generalization. The Cohere2 binding, corpus, curation and evaluation are new hackathon work; Particle Sliders core existed beforehand. Tiny Aya and derived weights are CC-BY-NC-4.0 for noncommercial use.

Word count: 305.

## Account fields

The owner has supplied the registration email and confirmed Discord membership.
Personal form fields are intentionally kept out of this public packet. The
registration's Hugging Face username matching `ntc-ai` remains an owner credit
check. No further form action is requested from an agent.
