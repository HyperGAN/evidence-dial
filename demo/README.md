---
title: Evidence Dial
emoji: 🎚️
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
short_description: Inspect learned behavior, prompting and regressions
---

# Support with independent judgment

Particle Sliders' Lexsi hackathon entry, **Track 1, section 7: Preference and style alignment**.

A recorded-output explorer for a raw ParticleGAN adapter on frozen Tiny Aya Global 3.35B. The target is sympathetic support with independent judgment. Compare OFF, strengths0.5/1/2, and a fixed behavioral instruction on identical held-out questions. Every displayed answer is an actual saved native generation. There is no new model call or paid API request when using this page.

| Condition | Mean behavior score,32 held-out questions |
| --- | ---: |
| OFF | 0.588828 |
| Prompt-only | 0.612266 |
| PG strength1, selected on DEV | 0.599609 |
| Same checkpoint, strength0.5, descriptive | 0.600781 |
| Same checkpoint, strength2, descriptive | 0.520234 |

The selected adapter's paired change from OFF is +0.010781, with an individual95% question-bootstrap interval [−0.012188,+0.034687]. Improvement over OFF is uncertain; no advantage over prompting or SFT is established. Strength2 regresses. Scores are model judgments, not accuracy percentages or human preference rates. A small synthetic corpus, one training seed, shared curation/evaluation judge and64-token limit restrict the conclusions. All32 cases and regressions remain inspectable.

- [Thorough README, abstract, method, complete results and limitations](https://github.com/HyperGAN/evidence-dial)
- [Behavior adapter](https://huggingface.co/ntc-ai/evidence-dial-behavior-particle)
- [Training data, all predictions, logs, checkpoints and audits](https://huggingface.co/datasets/ntc-ai/evidence-dial-behavior)
- [Public model collection](https://huggingface.co/collections/ntc-ai/evidence-dial-models-and-adapters-6af28e2624ee9d502264a99f)
- [Public dataset collection](https://huggingface.co/collections/ntc-ai/evidence-dial-datasets-and-benchmark-6af2746d22e8aefe32cafb87)
- [Archived development preview](https://ntc-ai-evidence-dial.static.hf.space/behavior-preview.html)
- [Earlier grounding explorer](https://ntc-ai-evidence-dial.static.hf.space/grounding.html)

Earlier QA evidence: PG and SFT both174/192 on development; on sealed256-question confirmation, all five preselected PG arms221–229, all four SFT controls230–234, OFF142. This is a development tie and a held-out gap to SFT.

Tiny Aya and derived weights/states are CC-BY-NC-4.0 for noncommercial use. Upstream code licenses apply. Live artifact/content verification is separate from a fresh-machine reproduction or browser interaction check.
