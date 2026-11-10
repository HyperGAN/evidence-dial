# Evidence Dial

Particle Sliders' Lexsi hackathon project: a learned grounding control for
`CohereLabs/tiny-aya-global`. The protocol and exact workflow are in
[`docs/reproduce.md`](docs/reproduce.md).

**Status: Particle training in progress; no holdout result yet.** The required
AlignTune LoRA completed128 updates (64 cases ×4 exposures) and scored128/128
on development, versus OFF86/128: paired gain+0.328125, family95% interval
[0.2734375,0.3828125]. This is a supervised development result, not a Particle
Sliders win or an untouched holdout result.

Development headroom is measured: identical-prompt OFF 86/128
(67.1875%); private answer-informed teacher 112/128 (87.5%). This teacher
comparison is a feasibility probe, not an adapter benchmark. The Cohere2 native
checks passed, including initial ON=OFF, all2,593 connected trainable tensors,
tensor-exact FULL0 restoration, and native-only RAW generation. Evidence is in
[`artifacts/evidence`](artifacts/evidence). **G0.012 stopped at step32** after a
qualified catastrophic check found degenerative `CONF` repetition. Its saved
FULL/RAW checkpoints stay intact and cannot resume or enter selection. G0.004
completed128 updates. Both nominal strengths scored0/128 on development. Its
old strength2 check mistook a permitted64-token cutoff for catastrophe; that
record is retained with an explicit scope audit. The new instrument passed
FULL128 but flagged a truncated mathematical fragment outside its established
coverage. Qualification remains uncertain and G0.004 is blocked for promotion.
G0.001 is completing its declared128 updates from its qualified step64 state. This is separate from effect
ranking and is not a zero score for an unmeasured RAW128.

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

The primary result is paired strict success ON minus OFF on 256 untouched
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
The optional closed-API ROI comparison has no configured credential and is not
claimed. The final benchmark, particle adapter, article and demo are pending.
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
the declared static explorer fallback is being prepared.

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
