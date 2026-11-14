---
title: Evidence Dial
emoji: 🎚️
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
short_description: Complete answers, learned steering and regressions
---

# Support with independent judgment — complete answers

Particle Sliders' Lexsi hackathon entry, **Track 1, section 7: Preference and style alignment**.

The main page shows **all 80 complete responses on 16 development questions** from the published 32-update adapter, with OFF, three strengths, and prompt-only. Every response ends at the model's native end token. Generation has a 4,096-token safety cap, and incomplete responses are not scored. The earlier 64-token budget cut off useful answers and materially changed the judgments.

Fresh training on **138 context pairs across 37 training questions**, from step 0 through **1,024 updates**, is running on both GPUs. Its completed results and fresh-question confirmation will be added later. This page remeasures the existing 37-pair, 32-update model; it does not show the new model.

| Fixed old checkpoint, complete-answer DEV remeasurement | Mean behavior score / 1 | Mean generated tokens |
| --- | ---: | ---: |
| OFF | 0.662031 | 401.75 |
| Prompt-only | 0.666563 | 60.44 |
| Strength 0.5, descriptive | 0.684219 | 256.44 |
| Strength 1, previously selected on the old protocol | 0.677500 | 111.19 |
| Strength 2, descriptive | 0.574219 | 48.44 |

Strength 1 minus OFF is +0.015469 with an individual paired 95% question-bootstrap interval [−0.041875, +0.075004]. Its change from prompt-only is +0.010937 [−0.027031, +0.050312]. These are uncertain development differences, not a demonstrated held-out advantage. Strength 2 regresses. The display reports all three fixed strengths without selecting another winner from these results.

The same questions and every saved original token prefix are independently verified across the old cutoff and new complete-answer versions. Scores are Jev model judgments, not accuracy percentages or human preference rates. The small synthetic corpus, one training seed, shared curation/evaluation judge, and first-state supervision constrain the conclusions.

- [Thorough README, method, results, and limitations](https://github.com/HyperGAN/evidence-dial)
- [Published behavior adapter](https://huggingface.co/ntc-ai/evidence-dial-behavior-particle)
- [Data and evaluation archive](https://huggingface.co/datasets/ntc-ai/evidence-dial-behavior)
- [Public model collection](https://huggingface.co/collections/ntc-ai/evidence-dial-models-and-adapters-6af28e2624ee9d502264a99f)
- [Public dataset collection](https://huggingface.co/collections/ntc-ai/evidence-dial-datasets-and-benchmark-6af2746d22e8aefe32cafb87)
- [Archived 64-token held-out comparison](https://ntc-ai-evidence-dial.static.hf.space/short-answer-holdout.html)
- [Earlier development preview](https://ntc-ai-evidence-dial.static.hf.space/behavior-preview.html)
- [Earlier grounding explorer](https://ntc-ai-evidence-dial.static.hf.space/grounding.html)

The older sealed 256-question QA study remains separate: PG and SFT each 174/192 on DEV; PG 221–229/256, SFT 230–234/256, OFF 142/256 on test. Development tie and held-out SFT gap, not parity.

This is a recorded-output explorer: no new model or paid API calls occur in the browser. Tiny Aya and derived weights/states retain CC-BY-NC-4.0. Browser interaction QA is separate from file/content verification.
