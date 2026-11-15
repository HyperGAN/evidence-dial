---
license: cc-by-nc-4.0
language:
- en
tags:
- particlegan
- preference-alignment
- internal-states
- cohere
configs:
- config_name: promptfix37_questions
  data_files:
  - split: train
    path: promptfix37-h1024-v4/questions/train.jsonl
  - split: validation
    path: promptfix37-h1024-v4/questions/dev.jsonl
  - split: test
    path: promptfix37-h1024-v4/questions/confirmation64.jsonl
  default: true
- config_name: confirmation_answers
  data_files:
  - split: test
    path: data/confirmation.jsonl
- config_name: training_contexts
  data_files:
  - split: train
    path: data/training-contexts.jsonl
- config_name: questions
  data_files:
  - split: train
    path: questions/train.jsonl
  - split: validation
    path: questions/dev.jsonl
  - split: test
    path: questions/holdout.jsonl
- config_name: mining
  data_files:
  - split: train
    path: curation/attempts-1024.jsonl
- config_name: complete-answer-dev
  data_files:
  - split: validation
    path: complete-answer-DEV-remeasurement-v1/predictions.jsonl
- config_name: expanded-fullanswer-confirmation
  data_files:
  - split: test
    path: expanded138-h1024-v2/evaluation/confirmation/combined.jsonl
- config_name: promptfix37_confirmation
  data_files:
  - split: test
    path: promptfix37-h1024-v4/evaluation/confirmation/combined.jsonl
---

# Particle Sliders for Tiny Aya — data, outputs, and experiment records

[Demo: all 64 examples](https://huggingface.co/spaces/ntc-ai/tiny-aya-particle-sliders) · [Article preview](https://huggingface.co/ntc-ai/tiny-aya-particle-sliders/blob/main/ARTICLE.md) · [Code](https://github.com/HyperGAN/tiny-aya-particle-sliders) · [Weights](https://huggingface.co/ntc-ai/tiny-aya-particle-sliders) · [Dataset](https://huggingface.co/datasets/ntc-ai/tiny-aya-particle-sliders) · [Model collection](https://huggingface.co/collections/ntc-ai/particle-sliders-for-tiny-aya-models-and-adapters-6af28e2624ee9d502264a99f) · [Dataset collection](https://huggingface.co/collections/ntc-ai/particle-sliders-for-tiny-aya-datasets-and-evidence-6af2746d22e8aefe32cafb87)

**A complete worked example of turning a written quality criterion into an adjustable ParticleGAN adapter.** The released criterion is sympathy with independent judgment. Other criteria need their own curation, training, and evaluation.

[Demo: all 64 examples](https://huggingface.co/spaces/ntc-ai/tiny-aya-particle-sliders) · [Winning adapter](https://huggingface.co/ntc-ai/tiny-aya-particle-sliders) · [Code and full evaluation](https://github.com/HyperGAN/tiny-aya-particle-sliders) · [Method and equations](https://huggingface.co/ntc-ai/tiny-aya-particle-sliders/blob/main/METHOD.md)

## Start with the current study

The current release is **`promptfix37-h1024-v4/`**. This repository also retains earlier experiments; they are historical evidence, not additional training data for the released adapter. The default viewer now opens the current question splits.

| What you want | Config or file | Contents |
| --- | --- | --- |
| Current questions | `promptfix37_questions` (default) | 64 TRAIN / 16 validation (DEV) / 64 fresh test questions |
| Easy-to-read complete answers | `confirmation_answers` | 384 rows: 64 questions × six fixed conditions |
| Approved training contexts | `training_contexts` | 37 rows from 37 distinct TRAIN questions, including the chosen target system prompt and original/target scores |
| Full original answer records | `promptfix37_confirmation` | Identical 384 answers with native token IDs, AuditKIT metadata, and judgment provenance |
| Frozen state tensors | [`approved37-first-states.pt`](https://huggingface.co/datasets/ntc-ai/tiny-aya-particle-sliders/blob/main/promptfix37-h1024-v4/training-pairs/approved37-first-states.pt) | Neutral/positive first-answer hidden-state pairs for training |
| Exact pair construction | [`training-pairs/index.json`](https://huggingface.co/datasets/ntc-ai/tiny-aya-particle-sliders/blob/main/promptfix37-h1024-v4/training-pairs/index.json) | Tokenized input pairs, target prompts, chosen source states, and hashes |
| Model selection | [`audits/development-search-closure.json`](https://huggingface.co/datasets/ntc-ai/tiny-aya-particle-sliders/blob/main/promptfix37-h1024-v4/audits/development-search-closure.json) | All 24 declared DEV settings and the locked winner |
| Finished-output comparison | [`audits/fresh-confirmation-comparison.json`](https://huggingface.co/datasets/ntc-ai/tiny-aya-particle-sliders/blob/main/promptfix37-h1024-v4/audits/fresh-confirmation-comparison.json) | Six conditions, paired gains, intervals, and completion checks |
| Run logs and saved states | [`training/`](https://huggingface.co/datasets/ntc-ai/tiny-aya-particle-sliders/tree/main/promptfix37-h1024-v4/training) | Two completed 1,024-update runs, raw/native/full states, plans, and logs |
| Original curation and sources | [`mining-states/`](https://huggingface.co/datasets/ntc-ai/tiny-aya-particle-sliders/tree/main/promptfix37-h1024-v4/mining-states) and [`provenance/`](https://huggingface.co/datasets/ntc-ai/tiny-aya-particle-sliders/tree/main/promptfix37-h1024-v4/provenance) | All candidate context captures and retained first judgments/source observations |

The two flat views under `data/` are convenience exports. They preserve the original answer text and scores exactly; the immutable study records above remain the evidence source.

## Load the useful views

```python
from datasets import load_dataset

repo = "ntc-ai/tiny-aya-particle-sliders"
questions = load_dataset(repo, "promptfix37_questions")
contexts = load_dataset(repo, "training_contexts", split="train")
answers = load_dataset(repo, "confirmation_answers", split="test")

assert [len(questions[s]) for s in ["train", "validation", "test"]] == [64, 16, 64]
assert len(contexts) == 37 and len(answers) == 384
```

Use a pinned repository revision for reproducible downloads. PyTorch files require the custom native runtime described in the model card; the JSONL views work independently of it.

## How the data was made

Questions are English, synthetic everyday scenarios, not real conversations or a human preference dataset. They come from related scenario families. TRAIN, DEV, and fresh confirmation have disjoint IDs and exact question text; family-level independence is not claimed.

The quality prompt is the **judge**. Its fixed criterion asks for attentive sympathy, useful support, and independent judgment. [Exact Jev instructions and five anchors](https://huggingface.co/ntc-ai/tiny-aya-particle-sliders#quality-criterion-used-for-this-example).

Target system prompts are **candidate ways to elicit that quality**. They can differ between examples. For every TRAIN question, the frozen Tiny Aya Global 3.35B model produced an OFF answer and answers under three unrestricted target prompts. All 256 candidate/control answers finished at native EOS. Jev (`typesafe/jev-1.14` through NanoGPT) scored the complete answers.

We retained at most one target per question: the best context scoring at least 0.75 and strictly above its matched OFF answer. This yields 37 contexts from 37 TRAIN questions. The 27 other questions supplied no accepted pair. The target selection does not use DEV or confirmation answers.

Neutral and positive inputs retain the same user question and empty continuation history. Only the positive input gets the behavioral system prompt; the model's normal chat preamble remains. The target tensor is the final hidden vector immediately before the first answer token, dimension 2,048. Generated response tokens establish which context qualifies; they are not an added SFT training loss. Training does not differentiate through Jev.

## Splits and selection

- **TRAIN: 64 questions.** Only the approved 37 contexts supply training targets.
- **Validation/DEV: 16 questions.** Selects among two learning-rate runs × four checkpoints × three strengths = 24 settings.
- **Fresh test: 64 questions.** Generated and evaluated after the checkpoint/strength choice was locked; it supplied no training targets and was not used to reselect the winner.

Both fresh runs completed 1,024 updates, batch 8. The winner was the lower adapter rate 0.0000625, step 1,024, strength 1. Model-weight EMA is disabled; saved and selected weights are raw.

## Answer fields and scores

`confirmation_answers` has one row per `(id, arm)`. `question` and `output` contain the complete strings. `effect` is the fixed five-anchor Jev score divided by four, on a 0–1 scale. `sympathy_probability` and `sycophancy_probability` are separate Jev yes/no probabilities, not the combined score. `termination` is `native_eos` in all 384 rows. `output_tokens` counts the retained generated token IDs.

`adapter_strength`, `has_behavioral_system_prompt`, and `selected_on_dev` describe each condition. The arms are original OFF, instruction-only control, selected adapter strengths 0.25 / 0.5 / 1, and the earlier adapter as a fixed reference. Only the selected current strength-1 arm is the primary preselected result. Intermediate strengths and the previous checkpoint are descriptive comparisons.

`training_contexts` includes the chosen `target_system_prompt`, `quality_score`, `original_score`, and positive/neutral state paths and SHA-256 hashes. State paths are relative to this dataset repository. Use `hf_hub_download(..., repo_type="dataset")` to retrieve them. Historical absolute paths in the original records are provenance, not runnable paths on another machine.

## Current result and limits

| Same 64 fresh questions | Mean Jev behavior score |
| --- | ---: |
| Original Tiny Aya | 0.664180 |
| Selected adapter, strength 1 | 0.732656 |

Paired gain **+0.068477**, individual 95% question-bootstrap interval **[+0.051211, +0.085392]**. These are model judgments, not accuracy percentages or human preference rates. The same judge curates and evaluates. There is one training seed, related synthetic scenarios, first-answer-state supervision, and no matched behavior-SFT or human study. Sympathy rises; the separate sycophancy diagnostic does **not** improve over OFF. The interval does not establish independence between scenario families. Fresh-machine replication is unverified.

## Earlier studies remain available

`questions`, `mining`, `complete-answer-dev`, and `expanded-fullanswer-confirmation` are earlier configurations. `expanded138-h1024-v2/` retains the larger earlier training study and its uncertain/negative confirmation. Earlier short-answer measurements, incomplete-output corrections, QA comparisons, first judgments, logs, and checkpoints remain preserved. They are not merged into the current TRAIN split. See the [full evaluation record](https://github.com/HyperGAN/tiny-aya-particle-sliders/blob/main/EVALUATION.md) and the two public collections for scope.

## License, attribution, and intended use

Tiny Aya-derived weights and hidden states retain **CC-BY-NC-4.0**; upstream software keeps its own licenses. This research/example preference dataset can contain unsupported assumptions and poor advice, including in higher-scoring answers. No real private conversations are included. Synthetic authorship, model-generated answers, and model-based judgments are explicit throughout.

ParticleGAN and Particle Sliders predate this entry. CuratorKIT validates schema/deduplication; AuditKIT wraps generation and judging. The entry is Lexsi Track 1, section 7: preference and style alignment.

[Demo](https://huggingface.co/spaces/ntc-ai/tiny-aya-particle-sliders) · [Winning adapter](https://huggingface.co/ntc-ai/tiny-aya-particle-sliders) · [GitHub](https://github.com/HyperGAN/tiny-aya-particle-sliders) · [Method](https://huggingface.co/ntc-ai/tiny-aya-particle-sliders/blob/main/METHOD.md) · [Model collection](https://huggingface.co/collections/ntc-ai/particle-sliders-for-tiny-aya-models-and-adapters-6af28e2624ee9d502264a99f) · [Dataset collection](https://huggingface.co/collections/ntc-ai/particle-sliders-for-tiny-aya-datasets-and-evidence-6af2746d22e8aefe32cafb87)
