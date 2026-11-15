---
title: Particle Sliders for Tiny Aya
emoji: 🎚️
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
short_description: Complete answers, learned steering and regressions
---

# Particle Sliders for Tiny Aya — sympathy with independent judgment

**A learned behavior slider for Tiny Aya Global 3.35B.** This small native
Particle Sliders adapter steers ordinary answers toward sympathy with independent
judgment. Keep the question unchanged and adjust the learned behavior's strength;
no additional behavioral instruction is needed at inference.


Lexsi hackathon, Track 1 section 7: Preference and style alignment.

On 64 fresh synthetic questions, the DEV-selected ParticleGAN adapter scores
**0.732656**, versus **0.664180 OFF** and **0.697578 prompt-only**. These are
fixed Jev behavior scores on a0–1 scale, not accuracy or human preference rates.
All384 complete native-EOS answers are included in this recorded-output explorer.

Choose a question and move among OFF and recorded strengths0.25/0.5/1. The
ordinary question remains unchanged; the learned adapter adds no behavioral
instruction at inference. The separate prompt-only panel shows that control.
The earlier expanded-study adapter remains available under checkpoint details.

The preselected adapter's paired gain overOFF is+0.068477, individual95%interval
[+0.051211,+0.085392]; overprompt-only it is+0.035078
[+0.013358,+0.058320]. All choices were locked before confirmation. Sympathy
judgment rises, while sycophancy judgment does not fall versusOFF
(0.088594 vs0.086094). One seed, related synthetic scenario families and a
shared curation/evaluation judge limit the conclusions. No matched behavior-SFT
or human preference study was performed.

Sample scores appear above each answer. ✅ Green means strictly higher than
matched original; ❌ red means equal or lower. These marks refer only to the
frozen Jev behavior judgment.

The Space performs no new inference or paid API calls. Complete earlier demo
pages and their data remain preserved. The main README includes the method,
all24 DEV settings, full results, real regressions, source/provenance and limits.

- [Project source and thorough README](https://github.com/HyperGAN/tiny-aya-particle-sliders)
- [Winning native adapter](https://huggingface.co/ntc-ai/evidence-dial-behavior-particle)
- [All data, checkpoints and original judgments](https://huggingface.co/datasets/ntc-ai/evidence-dial-behavior/tree/main/promptfix37-h1024-v4)
- [Model collection](https://huggingface.co/collections/ntc-ai/evidence-dial-models-and-adapters-6af28e2624ee9d502264a99f)
- [Dataset collection](https://huggingface.co/collections/ntc-ai/evidence-dial-datasets-and-benchmark-6af2746d22e8aefe32cafb87)

Separate earlier QA evidence tied PG and SFT at174/192 onDEV; held-out256-question
results favored SFT. This behavior study does not establish SFT parity or SOTA.

[Direct winning weights](https://huggingface.co/ntc-ai/evidence-dial-behavior-particle/resolve/b7e7f256035b9eb8d4ded4d01af52effff2e51fc/adapter-native.pt?download=true) · [Winner record](https://huggingface.co/ntc-ai/evidence-dial-behavior-particle/blob/main/winner.json) · [Method and equations](https://huggingface.co/ntc-ai/evidence-dial-behavior-particle/blob/main/METHOD.md)
