"""Write the final developer entry from the completed locked comparison."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/behavior-promptfix-20261114-v4'
PREFIX = 'promptfix37-h1024-v4'
DATA = 'https://huggingface.co/datasets/ntc-ai/evidence-dial-behavior'
DEMO = 'https://ntc-ai-evidence-dial.static.hf.space/'
OLD_README = 'https://github.com/HyperGAN/evidence-dial/blob/bdecc3d725e8ec80271a2a292b9b59158e00f9e2/README.md'


def pin(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    audit_path = BASE / 'locked-fresh64-v4/independent-completion-audit-first.json'
    audit = json.loads(audit_path.read_text())
    closure = json.loads((BASE / 'development-search-closure-first.json').read_text())
    gallery = json.loads((ROOT / 'demo/promptfix-confirmation.json').read_text())
    assert audit['all_answers_native_EOS'] and audit['no_weight_EMA'] and not audit['new_test_reselection']
    assert audit['total_test_answers'] == 384 and len(gallery['cases']) == 64
    selected = next(a for a in audit['arms'] if a['primary_selected'])
    off = next(a for a in audit['arms'] if a['name'] == 'off')
    prompt = next(a for a in audit['arms'] if a['name'] == 'prompt-only')
    comparisons = {c['baseline']: c for c in audit['comparisons'] if c['primary_selected']}
    assert comparisons['off']['mean_difference'] > 0
    candidate = closure['display_checkpoint_representative']
    assert candidate['name'] == selected['name'] and len(closure['all24_arms']) == 24
    model = gallery['default_model_id']
    def question(suffix):
        return next(c for c in gallery['cases'] if c['id'].endswith('-' + suffix))
    good, bad = question('021'), question('031')
    labels = {'off': 'OFF — unmodified base', 'prompt-only': 'Unrestricted prompt-only',
        selected['name']: 'PG strength 1 — selected on DEV',
        'PG-GPU1-step1024-alpha0.25': 'PG strength 0.25 — descriptive',
        'PG-GPU1-step1024-alpha0.5': 'PG strength 0.5 — descriptive',
        'previous-expanded138-PG-GPU1-step256-alpha1': 'Earlier expanded-study adapter — fixed reference'}
    rows = ['| Frozen condition, 64 fresh questions each | Mean behavior score | Change from OFF |',
            '| --- | ---: | ---: |']
    for a in audit['arms']:
        rows.append(f'| {labels[a["name"]]} | {a["mean_effect"]:.6f} | {a["mean_effect"] - off["mean_effect"]:+.6f} |')
    table = '\n'.join(rows)
    contrast_rows = ['| Selected PG compared with | Mean paired gain | Individual 95% bootstrap interval | Wins / ties / losses |',
                     '| --- | ---: | --- | --- |']
    for key, label in [('off', 'OFF'), ('prompt-only', 'Prompt-only'),
            ('previous-expanded138-PG-GPU1-step256-alpha1', 'Earlier adapter')]:
        c = comparisons[key]
        low, high = c['individual_paired_bootstrap95']
        contrast_rows.append(f'| {label} | {c["mean_difference"]:+.6f} | [{low:+.6f}, {high:+.6f}] | ' + ' / '.join(map(str, c['wins_ties_losses'])) + ' |')
    contrasts = '\n'.join(contrast_rows)
    diagnostics = ['| Condition | Sympathy judgment | Sycophancy judgment | Mean answer tokens |',
                   '| --- | ---: | ---: | ---: |']
    for a, label in [(off, 'OFF'), (prompt, 'Prompt-only'), (selected, 'Selected PG')]:
        diagnostics.append(f'| {label} | {a["mean_sympathy_probability"]:.6f} | {a["mean_sycophancy_probability"]:.6f} | {a["mean_tokens"]:.1f} |')
    diagnostics = '\n'.join(diagnostics)
    dev_rows = ['| G learning rate | Updates | Strength 0.25 | Strength 0.5 | Strength 1 |',
                '| ---: | ---: | ---: | ---: | ---: |']
    for gpu in [0, 1]:
        for step in [256, 512, 768, 1024]:
            arms = [a for a in closure['all24_arms'] if a['gpu'] == gpu and a['step'] == step]
            assert len(arms) == 3
            arms.sort(key=lambda a: a['strength'])
            dev_rows.append(f'| {arms[0]["g_lr"]:g} | {step:,} | ' + ' | '.join(f'{a["mean_effect"]:.6f}' for a in arms) + ' |')
    dev_table = '\n'.join(dev_rows)
    abstract = f'''Evidence Dial turns written response preferences into an adjustable adapter for frozen Tiny Aya Global 3.35B. Its target is sympathetic, constructive support with independent judgment. For each of 64 TRAIN questions, the unmodified model generates complete responses under three unrestricted behavioral instructions and a neutral control. NanoGPT Jev filters successful contexts; one best qualifying context per question yields 37 paired first-answer hidden states. Neutral and positive inputs retain the same question and empty continuation history, while only positive receives the behavioral instruction.

The unchanged ParticleGAN 0.9.5 formulation trains a routed adapter, critic and particle bank while the backbone stays frozen. Two fresh learning-rate cohorts complete 1,024 updates each, with raw weights and no weight EMA. Selection uses finished intended effect ON minus matched OFF across all 24 declared DEV settings. At inference, the adapter retains the ordinary question, adds no behavioral prompt, and exposes an intensity control.

On 64 fresh synthetic questions, the preselected adapter scores {selected['mean_effect']:.4f}, versus {off['mean_effect']:.4f} OFF and {prompt['mean_effect']:.4f} prompt-only. Its gains are {comparisons['off']['mean_difference']:+.4f} and {comparisons['prompt-only']['mean_difference']:+.4f}; both individual paired bootstrap intervals exclude zero. All 384 comparison answers reach native EOS. Sympathy judgments rise, while the separate sycophancy diagnostic does not fall versus OFF. Earlier negative results remain public. One training seed, related synthetic scenario families, a shared curation/evaluation judge and first-state supervision limit the conclusions.'''
    text = f'''# Evidence Dial

**Lexsi hackathon — Track 1, section 7: Preference and style alignment.**

Teach a frozen model a written response preference, then control its strength
while keeping the ordinary user question unchanged. The demonstrated target is
**sympathy with independent judgment**: acknowledge feelings, question an
unsupported conclusion tactfully, and support a reasonable boundary.

**Fresh result:** the DEV-selected ParticleGAN adapter scores **{selected['mean_effect']:.4f}**,
versus **{off['mean_effect']:.4f} OFF** and **{prompt['mean_effect']:.4f} prompt-only**, on 64 new questions.
This is a fixed Jev behavior score on a 0–1 scale, not accuracy or a human
preference percentage. All six conditions and all 384 complete answers are public.

[Recorded complete-answer demo](https://huggingface.co/spaces/ntc-ai/evidence-dial) ·
[Raw model adapter](https://huggingface.co/ntc-ai/evidence-dial-behavior-particle) ·
[Data, checkpoints and audits]({DATA}/tree/main/{PREFIX}) ·
[Model collection](https://huggingface.co/collections/ntc-ai/evidence-dial-models-and-adapters-6af28e2624ee9d502264a99f) ·
[Dataset collection](https://huggingface.co/collections/ntc-ai/evidence-dial-datasets-and-benchmark-6af2746d22e8aefe32cafb87)

## Abstract

{abstract}

## Where this adds value

- **Preferences become internal targets.** A positive context can specify how to reason, search, or respond; the neutral context keeps the same question. The demonstrated model learns from the frozen model's prompt-induced hidden states. Private reasoning quality is not directly measured.
- **Different examples can use different successful instructions.** Curation chooses among three target prompts per question. No single behavioral prompt is imposed on every training example.
- **The learned behavior has a runtime control.** One adapter receives ordinary questions and supports strength adjustment. The recorded demo exposes OFF, 0.25, 0.5 and 1; it never interpolates answer text. Learning another preference requires new curation and training.
- **The comparison includes what prompting already achieves.** The selected adapter improves the fixed judge's mean over both OFF and the unrestricted prompt control on fresh questions. Source, original judgments, failures, regressions and earlier studies remain inspectable.

This is an evaluated alternative training objective and an integration pipeline.
The study does not establish a general efficiency advantage over SFT, and the
behavior experiment has no matched behavior-SFT baseline.

## Fresh 64-question comparison

{table}

All checkpoint and strength choices were frozen before these answers were
generated. Strength 1 is the unique winner of the completed 24-cell DEV search;
0.25 and 0.5 are descriptive display conditions. The earlier adapter is a fixed
reference selected in the previous study. These results do not select a new
winner on the confirmation set.

{contrasts}

Intervals use 10,000 paired-question bootstrap resamples, seed 20261114.
They are individual intervals, with no simultaneous-coverage or independent
semantic-family claim. Questions receive equal weight.
[Independent comparison audit]({DATA}/blob/main/{PREFIX}/audits/fresh-confirmation-comparison.json).

### Separate behavioral diagnostics

{diagnostics}

Sympathy and sycophancy are mean Jev probabilities, reported separately from
the primary effect score. Sympathy rises. Sycophancy is **0.0886 versus 0.0861
OFF**, so this experiment does not demonstrate a reduction on that diagnostic.
These diagnostics never select checkpoints or break ties.

## Inspect the behavior

The explorer contains every fresh question, both controls, every locked
strength and the earlier adapter reference. The question remains unchanged;
the prompt-only panel shows the additional instruction explicitly. Replies
are saved complete native generations, with token counts and termination.
It performs no new inference or paid API calls.

The following are post hoc illustrations. They do not select or rerate a model.

**Unsupported conclusion:** {good['question']}

Jev scores OFF **{good['off']['effect']:.4f}**, prompt-only **{good['prompt_only']['effect']:.4f}**,
and selected PG **{good['models'][model]['1']['effect']:.4f}**. The adapter explicitly advises
against assuming the friend's motive, but still produces a generic list and
speculative possibilities. [Open this complete comparison]({DEMO}?case={good['id']}).

<details>
<summary>All three complete recorded answers</summary>

### OFF

{good['off']['output'].strip()}

### Prompt-only

{good['prompt_only']['output'].strip()}

### Selected PG, strength 1

{good['models'][model]['1']['output'].strip()}

</details>

**A real regression:** {bad['question']}

OFF scores **{bad['off']['effect']:.4f}** and PG **{bad['models'][model]['1']['effect']:.4f}**.
The adapter invents a change of plans and treats the other speaker as the
decision-maker. A warmer tone has not fixed role consistency.
[Open this complete comparison]({DEMO}?case={bad['id']}).

<details>
<summary>Both complete recorded answers</summary>

### OFF

{bad['off']['output'].strip()}

### Selected PG, strength 1

{bad['models'][model]['1']['output'].strip()}

</details>

The largest judged regression is the bedtime-story question: PG 0.5825 versus
OFF 0.7100. It remains visible alongside the gains.
[Complete case]({DEMO}?case=behavior-promptfix-fresh64-005).

## How it works

```mermaid
flowchart LR
  Q[Same TRAIN question] --> N[Frozen base: neutral context]
  Q --> P[Frozen base: behavioral context]
  P --> J[Jev filters complete target answers]
  N --> H[Paired first-answer hidden states]
  J --> H
  H --> G[ParticleGAN trains routed adapter]
  G --> A[Ordinary question plus runtime strength]
  A --> E[Complete output; AuditKIT and frozen judge]
```

1. Generate one OFF answer and three prompted answers for each of 64 TRAIN
   questions. Every candidate must finish at native EOS. Keep a positive target
   only when its normalized effect is at least 0.75 and exceeds matched OFF.
   Retain one highest-scoring target per question: **37 contexts / 37 questions**.
2. Capture the frozen final hidden state immediately before the first answer
   token. Positive and neutral share the question and empty continuation
   history. Only positive adds the behavior instruction; Aya's default
   preamble is preserved.
3. Train with original ConditionalPG09 Gaussian paired-error behavior,
   sigma 1, a rank-8 routed adapter and a 128×4 particle bank. The backbone is
   frozen. Initialization is scoped; saved states resume exactly. Model-weight
   EMA is disabled for adapter, critic and particle bank; raw weights are used
   for selection, exports and inference. Upstream scalar optimizer statistics
   remain unchanged.
4. Complete both 1,024-update cohorts and all declared DEV checks before
   selection. Rank only equal-question finished intended effect ON minus OFF.
   A separately qualified catastrophic-output check guards continuation;
   weak effect is not catastrophic failure. SSE is diagnostic, not a selector.
5. Lock every confirmation arm before inference. Generate greedily with a
   4,096-token safety cap; a cutoff is unfinished and unscored. Reuse the
   original successful native and Jev records rather than rerolling them.

The target is an internal-state distribution rather than a set of tokens to
imitate. There is no SFT warm start or text-imitation training loss in this
behavior experiment. [Meta Glimmer's paired-context method](https://github.com/HyperGAN/meta-glimmer-particle-slider/tree/813ea28dd37c156a23470d1ef9a4f1e418c581ac)
is the reference; ParticleGAN and Particle Sliders are pre-existing libraries.

## Training and DEV selection

Both cohorts start fresh and finish 1,024 updates, batch 8. G learning rates
are 0.00025 and 0.0000625; D is 0.018 and bank is 0.03. The original critic
has 16 SiLU hidden units and a 2,052-dimensional condition. The same training
seed is used; this is not a multi-seed comparison.

{dev_table}

Matched DEV OFF is 0.662031 and corrected prompt-only is 0.705156 on 16 questions.
The exact unique maximum is **G 0.0000625, step 1,024, strength 1: 0.713906**.
All 384 DEV answers are complete. Final strength-1 scores improve both cohorts'
earlier maxima, with mixed intermediate and other-strength results.

![All 24 declared DEV settings, relative to matched OFF](docs/figures/promptfix-DEV-training-progress-24cells-v5.png)

## What failed, and what changed

The earlier 138-context / 37-question study also completed both 1,024-update
runs. Its DEV-selected adapter scored **0.673906 versus 0.686719 OFF** on 32
fresh questions: gain **−0.012812**, individual 95% interval
**[−0.047266, +0.020781]**. It did not establish a base-model improvement.
[Complete earlier README and results]({OLD_README}) and
[all earlier full answers]({DEMO}expanded-confirmation.html) remain available.

The original target also imposed two or three short sentences. Removing that
unrelated restriction improved prompt-only DEV performance from 0.666563 to
0.705156, versus OFF 0.662031. New curation uses three unrestricted instructions
and one best context per admitted question. Twenty-six TRAIN questions overlap
the old cache; eleven newly qualify and eleven old questions are excluded.
Learning rates, target construction and question weighting also differ, so this
is not an isolated causal test of removing the sentence limit.

An offline check also found that AuditKIT's native run fingerprint omits
`experiment_name`. The qualified v5 wrapper binds the output namespace into
the model identity, preventing cross-directory reuse of same-arm labels.
Six saved-answer replays verified the fix without new model or judge calls.
Earlier failed checks and original sources are retained.

[Detailed setup diagnosis](docs/prompt-setup-diagnosis-v3.md).
A role-aware 46-question / 138-context derivative is prepared but **untrained**;
it contributes no result in this entry. Earlier short-answer studies and
operational failures remain archived; completed updates were not replayed.

## Separate capability check against SFT

The earlier QA study tied ParticleGAN and SFT at **174/192 on DEV (90.6%)**.
On the sealed 256-question test, five preselected PG arms scored **221–229**,
four SFT controls **230–234**, and OFF **142**. SFT performed better on that test.
This is a separate capability experiment; it does not establish SFT parity on
held-out data, a behavior-SFT win, or SOTA.
[Original QA tables and controls]({OLD_README}#separate-capability-check-against-sft).

## Reproduce and inspect

Recorded versions: Particle Sliders core 0.14.2 at
`63f6e3942340d8aadca816800d19fb1ac88d92a9`, ParticleGAN 0.9.5,
Transformers 4.57.6, CuratorKIT 0.1.21, AuditKIT 1.1.6 and PyTorch 2.14.1+cu130.
The pinned base is
`CohereLabs/tiny-aya-global@af89d219b53ed9b13b8a4645f9c8028973510324`.
It runs in BF16/SDPA on GPU. The experiment loader fixes a 22-GiB allocator
cap and expects a GPU larger than that cap. Shared-host run plans bind GPU
UUIDs, leases, memory floors and local paths; port those through a new execution
configuration. Fresh-machine end-to-end replication has not been verified.

```bash
git clone https://github.com/HyperGAN/evidence-dial
cd evidence-dial
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/hf download CohereLabs/tiny-aya-global --revision af89d219b53ed9b13b8a4645f9c8028973510324
```

These are custom native artifacts, not standard PEFT weights. Inside an active
bounded GPU runner, `resource` supplies its monitor's `check()` method:

```python
from huggingface_hub import hf_hub_download
from evidence_dial.behavior_inference_v5 import open_behavior_inference

artifact = hf_hub_download(
    "ntc-ai/evidence-dial-behavior-particle",
    "{PREFIX}/checkpoints/{candidate['artifact']['sha256'][:16]}/adapter-native.pt",
)
with open_behavior_inference(
    artifact, resource=resource,
    expected_sha256="{candidate['artifact']['sha256']}",
) as model:
    answer = model.respond(
        text="Your ordinary question", strength=1,
        capture_path="native-answer.json",
    )
print(answer["output"])
```

The model loads once per context and the owner makes no Jev calls. Captures
bind the exact artifact, question, decoder and strength. Unfinished output is
preserved and raises. For new curation/judgments, supply `NANOGPT_API_KEY`
locally; the route is NanoGPT `typesafe/jev-1.14`. Credentials are excluded.

| Entry point | Purpose |
| --- | --- |
| `src/evidence_dial/behavior_protocol_v1.py` | Target instructions and fixed Jev rubric |
| `tools/mine_behavior_promptfix_v4.py` | Complete target mining and frozen first-state captures |
| `tools/build_behavior_promptfix_cache_v4.py` | Independent curation and 37-pair cache |
| `tools/run_behavior_promptfix_training_v4.py` | Fresh training and complete quarter evaluations |
| `tools/audit_behavior_promptfix_quarter_v4.py` | Tokens, typed judgments, checkpoints, rates and no-EMA checks |
| `tools/close_behavior_promptfix_training_v4.py` | All-24-cell exact selection |
| `tools/confirm_behavior_promptfix_v5.py` | Exact pre-inference arm lock and fresh comparison |
| `tools/audit_behavior_promptfix_confirmation_v4.py` | Independent complete-answer comparison and paired intervals |
| `src/evidence_dial/behavior_inference_v5.py` | Complete-answer inference owner |
| `tools/export_behavior_promptfix_explorer_v4.py` | Every saved comparison answer for the demo |

## Public artifacts

- [Raw adapters and selection metadata](https://huggingface.co/ntc-ai/evidence-dial-behavior-particle/tree/main/{PREFIX}).
- [Questions, accepted states, all results, update journals and audits]({DATA}/tree/main/{PREFIX}). Public FULL/RAW/NATIVE milestones are at 0/256/512/768/1,024; every per-update FULL checkpoint is retained locally.
- [Native captures, original Jev records, execution receipts and source bodies]({DATA}/blob/main/{PREFIX}/provenance/observations-and-sources.tar.gz).
- [All 384 saved answers as portable JSON](demo/promptfix-confirmation.json).
- Both HF collections above contain public repositories. Earlier weights, datasets, logs and demo pages remain preserved.

## Limitations

- Only 64 fresh synthetic English questions, with broad scenario-family overlap. Exact questions differ from the previous 144; this is not an external benchmark or semantic-family-disjoint test.
- One training seed and the same curation/evaluation judge. No human preference study was performed; higher judge scores do not prove human-rated care or factual correctness.
- First-answer-state supervision does not directly teach later reasoning or guarantee role consistency. The regression examples remain important counterexamples.
- The separate sycophancy diagnostic did not improve over OFF. A larger strength does not guarantee a better answer on every case.
- Prompt construction, accepted questions, weighting and learning rate changed together. The study does not isolate one cause of the improvement.
- Individual bootstrap intervals are descriptive and provide no simultaneous coverage guarantee. All frozen conditions are reported; confirmation does not select another strength.
- No matched behavior-SFT baseline, demonstrated training-cost advantage, fresh-machine replication, or SOTA claim. The separate QA held-out gap favors SFT.
- Recorded-output Space, not a live inference service. Browser visual/click QA depends on an available browser host and is reported separately from content/hash verification.

## Toolkit use and attribution

[CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT) validates the synthetic
question schema and exact deduplication. [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT)
wraps native generation with the fixed Jev scorer. [AlignTune](https://github.com/Lexsi-Labs/aligntune)
supplies SFT baselines in the separate QA study; the behavior adapter uses
[ParticleGAN](https://github.com/HyperGAN/ParticleGAN) through
[Particle Sliders](https://github.com/HyperGAN/particle-sliders).

ParticleGAN, Particle Sliders and Meta Glimmer predate the entry. This project
adds the Tiny Aya integration, behavioral corpus/curation, paired-state pipeline,
bounded experiment harness, audits and complete-answer explorer. Tiny Aya-derived
weights and states retain **CC-BY-NC-4.0**; upstream code licenses remain in force.
[Submission packet](docs/submission-packet.md).
'''
    backup = ROOT / 'publication/promptfix-source-review-v4/README-before-final-first.md'
    assert not backup.exists()
    backup.write_bytes((ROOT / 'README.md').read_bytes())
    (ROOT / 'README.md').write_text(text)
    packet = f'''# Evidence Dial — hackathon submission packet

Team: **Particle Sliders**. **Track 1, section 7: Preference and style alignment.**
The owner completes the submission form; this repository claims no form-submission receipt.

| Submission field | Value |
| --- | --- |
| Main source and thorough README | https://github.com/HyperGAN/evidence-dial |
| Complete recorded-output demo | https://huggingface.co/spaces/ntc-ai/evidence-dial |
| Raw behavior adapter | https://huggingface.co/ntc-ai/evidence-dial-behavior-particle |
| Full data, checkpoints, predictions and audits | {DATA} |
| Public model collection | https://huggingface.co/collections/ntc-ai/evidence-dial-models-and-adapters-6af28e2624ee9d502264a99f |
| Public dataset collection | https://huggingface.co/collections/ntc-ai/evidence-dial-datasets-and-benchmark-6af2746d22e8aefe32cafb87 |

## Abstract

{abstract}

## Results and value

{table}

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
'''
    packet_path = ROOT / 'docs/submission-packet.md'
    packet_backup = ROOT / 'publication/promptfix-source-review-v4/submission-packet-before-final-first.md'
    assert not packet_backup.exists()
    packet_backup.write_bytes(packet_path.read_bytes())
    packet_path.write_text(packet)
    observation = BASE / 'final-entry-README-and-submission-packet-first.json'
    assert not observation.exists()
    observation.write_text(json.dumps({'state': 'final-entry-written-from-completed-confirmation',
        'source': pin(__file__), 'audit': pin(audit_path), 'README': pin(ROOT / 'README.md'),
        'submission_packet': pin(packet_path), 'previous_working_README': pin(backup),
        'previous_working_packet': pin(packet_backup), 'selected_on_DEV': candidate['name'],
        'new_model_or_Jev_calls': 0, 'published': False}, indent=2) + '\n')
    print(json.dumps({'README': pin(ROOT / 'README.md'), 'packet': pin(packet_path),
        'observation': pin(observation), 'new_model_or_Jev_calls': 0, 'published': False}))


if __name__ == '__main__':
    main()
