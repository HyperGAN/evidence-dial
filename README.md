# Evidence Dial

Particle Sliders' Lexsi hackathon project: a learned grounding control for
`CohereLabs/tiny-aya-global`. The protocol and exact workflow are in
[`docs/reproduce.md`](docs/reproduce.md).

**Problem statement — Track1, problem3:** answer from an organization's supplied
documents, say UNKNOWN when the answer is absent, and CONFLICT when equally
authoritative evidence disagrees. The experiment asks whether a continuous
particle adapter improves those finished answers, and where it beats a
supervised LoRA trained on the same source cases.

**Status: a Jev-curated successful-sample retry is training.**

The results below belong to the original answer-hinted target recipe. The corrected
workflow mines actual Tiny Aya successes, retains seed/token/hidden-state provenance,
and caches exact positive/neutral pairs. It retained48 of64 source cases; a same48-case
LoRA control and a fresh768-case confirmation are declared before evaluation. No new
quality win is claimed yet. See [the method and audit trail](docs/method-repair.md).
The [curated samples and paired-state archive](https://huggingface.co/datasets/ntc-ai/evidence-dial-curated-successes)
are public and verified. The [saved-output explorer](https://huggingface.co/spaces/ntc-ai/evidence-dial)
currently labels the complete original experiment; the corrected retry is pending.

| Original policy holdout | Strict successes |
| --- | ---: |
| OFF |169/256 (66.02%)|
| Selected particle G0.001 / strength1 |184/256 (71.88%)|
| Same particle / strength2 |173/256 (67.58%)|
| AlignTune supervised LoRA |256/256 (100%)|

Strength1 was selected on development before any holdout output. Its primary
paired gain over OFF is+5.86 percentage points, family95% interval[+1.17,+10.55].
LoRA is clearly stronger on this original test. The pipeline's ability to train
and improve over OFF is separate from comparative superiority over LoRA.

An independent audit reproduced all256 LoRA raw exact gold matches and verified
native input/output tokens, split isolation and the absence of gold metadata in
model prompts. It also found a weak benchmark: all16 question frames occur in
training and holdout, and DocumentB perfectly predicts CONFLICT across448 cases.
These results do not establish general grounding. See
[`docs/benchmark-audit.md`](docs/benchmark-audit.md).

At the owner's request,80 targeted cue checks and768 balanced cases were frozen
before any model output on them. One/two/three documents are balanced across
answer categories, with agreement, wrong-scope distractors, internal conflicts,
fresh values/wording and384 order pairs. The original900-case FaithEval transfer
benchmark is complete, alongside the cue audit and balanced test. All original scores and failures remain visible.
The separately authorized curated retry does not change the old selection.

| Original selected particle versus LoRA | OFF | Particle strength1 | LoRA | Particle − LoRA, pp ·95% interval |
| --- | ---: | ---: | ---: | --- |
| Policy256 |169|184|256|−28.13 [−32.81,−23.83]|
| FaithEval900 |323|337|375|−4.22 [−6.69,−1.74]|
| Cue audit80 |53|43|36|+8.75 [+3.75,+13.75]|
| Balanced stress768 |353|341|492|−19.66 [−23.96,−15.36]|

The original particle has a specific transfer advantage on FaithEval's
missing-evidence stratum:196/300 versus LoRA166/300, +10pp [5.37,14.57]. It loses
on counterfactual cases130/300 versus180/300 and inconsistent cases11/300 versus
29/300. These descriptive strata have unadjusted intervals. On the cue audit
it beats LoRA while both adapters hurt OFF. The balanced test favors LoRA
strongly. None of these results establishes broad particle superiority.

All three declared Particle rates were handled: G0.012 permanently stopped32
for qualified repetitive collapse; G0.004 completed128 but both nominal dev
strengths scored0/128 and mathematical-notation guard coverage remains uncertain,
so it is blocked; G0.001 completed128 with passing checkpoint and nominal-strength
checks. Native proof includes zero initial output/OFF equivalence,2593 connected
trainable tensors, exact FULL restore and RAW native-only generation. Teacher
headroom was112/128 versus OFF86/128; its cache was prepared once and retained.

The pipeline uses CuratorKIT schema checks, exact deduplication and exports;
Particle Sliders' shared routed projection and official ParticleGAN 0.9.5 game;
AlignTune's supervised training arm; and AuditKIT's callable/scorer interface.
The Cohere2 binding and all project datasets/evaluation are new hackathon work.
Particle Sliders is a pre-existing library.

## Protocol

The same frozen prompt, evidence, greedy decoding, native EOS and maximum 64
new tokens apply to every evaluated arm. OFF bypasses the adapter. Particle
strengths 1 and 2 use the same RAW weights. AlignTune is a separate supervised
LoRA trained on the same 64 gold examples. No weight EMA is permitted.

The declared primary result is paired strict success ON minus OFF on256
in-domain cases, with equal case weights and policy-family bootstrap intervals.
Development selects rate/strength before any holdout generation. A missing
generation is unmeasured. Teacher error and losses are diagnostics. A separately
qualified catastrophic-only output check governs continuation and promotion.

Frozen synthetic policy splits contain 64 training /128 development /256
holdout cases. Each split has equal answerable, missing, conflicting and
altered-fact cases. Fictional organizations/policy families are disjoint;
templates are shared. Every altered fact contradicts an everyday convention.
The upstream chat template uses 365 tokens before the task: actual policy
inputs, including private teacher information, fit within 505 tokens.

The frozen FaithEval transfer subset has 300 cases per stratum (900 total).
Eligibility precedes sampling; 338 overlength inputs are excluded without
truncation. This is a benchmark subset, not a leaderboard result. The full
1,500-row inconsistent-set audit could not verify its conservative structural
and exact-text control-construction check on483 rows. This does not prove that
483 source questions lack a semantically valid control. Full semantic support
validation was not performed, so matched FaithEval controls are omitted.
All exclusions and frozen IDs are retained in the data artifacts.

## Reproduce the completed stages

Python 3.11, Transformers 4.57.6 and the exact base revision in
[`configs/sources.lock.json`](configs/sources.lock.json) are required. Download
the gated base with an account that has access; credentials never go in code.
Kit Git revisions are pinned in that lock. Use a separate environment; the
local `.venv` inherits this host's Torch 2.14.1+cu130 installation.

```bash
python -m pip install --no-deps -e vendor/CuratorKIT -e vendor/AuditKIT -e .
python -m pytest tests -q
evidence-dial build-policy --out artifacts/data/policy-v1
evidence-dial freeze-faith --out artifacts/data/faith-v1
evidence-dial headroom --data artifacts/data/policy-v1 --out artifacts/runs/headroom-v3 --gpu 1
evidence-dial cache --data artifacts/data/policy-v1 --out artifacts/cache/teacher-v1 --headroom artifacts/runs/headroom-v3/gate.json --gpu 1
```

Build/freeze/cache commands refuse existing cohorts. Evaluation resumes by
stable identity and reuses completed native captures. Never rerun a completed
stage simply to recreate its paperwork. The `.venv`, vendor checkouts, gated
base weights and RAW/FULL model packets stay outside Git.

## Licences and limits

Tiny Aya and derived adapters are subject to the base's **CC-BY-NC-4.0**
non-commercial licence. CuratorKIT and AuditKIT retain their upstream
source-available licences; AlignTune retains its upstream licences. FaithEval
retains its original dataset terms. No gated base weights are redistributed.
The optional closed answer-model API comparison was not measured. NanoGPT is
used as the Jev curator, not as an answer-model comparison. Corrected results,
final particle-model publication and the final article/submission abstract are pending.
The owner will fill in the final hackathon form using the submission packet.

The AlignTune integration uses its pinned source `RaftTrainer` with measured
TRL0.25.1/Transformers4.57.6 compatibility; its latest all-extras dependency
bundle is not installed. No citation-loss mechanism is credited. The first
setup attempt failed before updates because of an inherited Arrow11 mismatch;
Arrow21 fixes it. Successful training and development evaluation exited0.

Five development OFF misses are answer-format differences (`cm/m`, a unit
omission, or an `Answer:` prefix). The frozen strict scorer is preserved;
reported protocol gains include these format changes. Per-case outputs expose
them. Fresh Particle training uses an explicitly experimental QR policy on
adapter leaves at the host constructor RMS, with zero up projections preserved;
this does not establish a general Transformer/LoRA initialization recommendation.

Public curated corpus: [ntc-ai/evidence-dial-policies](https://huggingface.co/datasets/ntc-ai/evidence-dial-policies).
It is ungated and its uploaded training bytes were checked against the frozen
local source. Public live Space creation returned HTTP402 (free ZeroGPU limit);
the declared static explorer fallback is live. Anonymous source/data bytes and
live-HTTP DOM behavior passed checks. The browser host is unavailable, so a
real browser visual check remains unmeasured.

The initial text checker also flagged the explicitly legal `UNKNOWN` label.
Two narrow audits preserve these old verdicts as out-of-contract, unmeasured
checks; they do not grant a healthy result. A distinct native-text instrument
passed8 frozen coverage cases with explicit greedy decoding. Original first
judgments and the earlier7-case proof are retained. New checkpoint checks are
required before promotion, and G0.012's real repetition failure stays blocked.
See `artifacts/evidence/native-text-qualification.json` and the source bundle.

Public [AlignTune adapter](https://huggingface.co/ntc-ai/evidence-dial-aligntune),
[model collection](https://huggingface.co/collections/ntc-ai/evidence-dial-models-and-adapters-6af28e2624ee9d502264a99f),
[dataset collection](https://huggingface.co/collections/ntc-ai/evidence-dial-datasets-and-benchmark-6af2746d22e8aefe32cafb87),
and [training archive](https://huggingface.co/datasets/ntc-ai/evidence-dial-training-archive)
are ungated. The archive preserves stopped and uncertain candidates with their
original guard records. Public model publication passed4 exact retained native
generation checks; this is implementation equivalence, not a holdout claim.

Fresh [cue-controlled grounding tests](https://huggingface.co/datasets/ntc-ai/evidence-dial-grounding-tests)
are public and verified byte-for-byte. Their manifests precede all new test
generation. The learned particle adapter has4,766,912 parameters, compared with
LoRA3,833,856; no footprint or inference-speed advantage is claimed.

The measured cohort uses one declared physical A6000, private sequencing,
explicit memory caps/floors and the original one-day deadline. One old
diagnostic-only50.98s phase omitted CUDA mapping: its physical device is
unverified. The record remains and the helper is repaired; no answer score or
selection depended on it. Cost accounting separates this exception and marks
legacy telemetry gaps rather than claiming complete one-device coverage.
