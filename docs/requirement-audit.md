# Plan requirement audit

Work in progress. This audit follows `docs/lexica-plan.md` in the parent
workspace. It does not claim a final benchmark or completed submission.

| Requirement | Evidence and current state |
| --- | --- |
| Track1/problem3, registered Particle Sliders | Submission packet records project/track; owner confirmed registration contact and Discord membership. |
| Pinned Tiny Aya Global3.35B | `configs/sources.lock.json`; exact base af89d219b53ed9b13b8a4645f9c8028973510324. No base weights redistributed. |
| Actual CuratorKIT → trainers → AuditKIT | `src/evidence_dial/data.py`, `aligntune_train.py`, `binding.py`, `evaluate.py`; actual pinned source paths, schema/dedup/exports, real RaftTrainer and callable scorers. |
| Policy64/128/256, family split first | `artifacts/data/policy-v1`;16/32/64 families with4 variants each, equal types; shared templates disclosed. |
| FaithEval eligibility before sampling | `artifacts/data/faith-v1/manifest.json`, exclusions and frozen-row metadata;900 frozen cases,338 overlength exclusions, no truncation. |
| Verify full inconsistent-set controls | Full1500-row conservative check unverified on483; semantic support not established. Declared fallback: omit matched FaithEval controls. |
| Original source reconstruction | `artifacts/evidence/faith-materialization.json`; actual reconstructed900-row file is byte-identical. Public metadata excludes passages/input IDs. |
| Headroom before binding/cache | `artifacts/evidence/headroom.json`: OFF86/128, private teacher112/128. |
| Teacher cache once, exact student prompt | Retained cache146 response positions from64 cases; captured original cache source in public source history. No teacher-informed evaluated arm. |
| Native binding proof | `artifacts/evidence/native-check.json`: zero initial output/OFF equality,2593 connected tensors, exact FULL restore, RAW native generation. |
| Scoped fresh initialization | Native proof and per-arm receipts preserve pretrained/tied weights, zero up outputs and restore states. Experimental adapter-leaf QR/R2 policy disclosed. |
| No model-weight EMA | All session/export receipts use `ema_decay=None`/RAW. Official scalar optimizer statistics preserved. |
| Three rates,128 updates, fixed D/bank | G.012 stopped32; G.004 completed128 but guard scope uncertain/blocked; G.001 completing128. Actual journals and RAW/FULL in training archive. No extra rates/horizon or checkpoint sweep. |
| Qualified catastrophe separate from effect | G.012 true repetition permanently stopped. Old cutoff/UNKNOWN flags preserved with narrow audits; distinct8-fixture native-text proof. G.004 mathematical-notation coverage remains uncertain and blocked. |
| No replay of completed updates/proofs | Immutable source/attempt captures, exact FULL resume; original native proof and teacher cache reused. Every G.001 update is journaled once. |
| Required AlignTune supervised arm |128 updates,256 exposures, each case4times; `artifacts/evidence/aligntune-training.json` and exposure audit. Dev128/128, gain+.328125; development only. |
| Matching prompts and native decoding | Greedy, EOS3,64 new tokens, train/dev≤512/eval≤2048; no evidence truncation. All128 LoRA/OFF dev native input sequences match. Full holdout audit pending. |
| RAW128 dev strength1 then2 | G.004 both measured0/128; G.001 pending. Extra strengths require a recorded declaration before outputs. |
| Sole effect selector, ties preserved | `benchmark.select` enforces complete nominal results, exact guard/checkpoint, failed/uncertain exclusions and numerical tie retention. Selection not yet frozen. |
| Holdout selection freeze before generation | No holdout generation yet. Pending full selection record and measured-throughput size declaration. |
| Primary256 and external transfer for all arms | Pending. Every arm must use the same frozen IDs; missing outputs stay unmeasured. |
| Equal-case effects, intervals, gains/losses | `report.py` uses10000 fixed-seed family/source-question cluster resamples. Final retained-output recomputation pending. |
| Per-type, answerable accuracy/abstention | Frozen diagnostics implemented. FaithEval matched-control rates unavailable because controls were not verified; disclose omission. |
| Published QA substring diagnostics | Separate diagnostic implementation; MC is explicitly a custom option-text substring check, not the unavailable published full MC script. No diagnostic ranks candidates. |
| Native response error and losses | Cached-target diagnostic and committed loss/exposure reporting implemented; final healthy-arm run pending. Failed/uncertain candidate diagnostics remain unmeasured. |
| Latency, tokens/sec, peak/GPU-hours incl teacher | Native records plus resource receipts. Legacy teacher time has artifact-timestamp estimate; peak unmeasured. Do not rerun cache/proofs to fill telemetry. Final cost report pending. |
| Optional closed-API ROI | Not run: no configured credential. Final report will give measured self-hosted generation cost at an explicit illustrative price, without a fabricated API comparison. |
| Public ungated model collection | AlignTune model published,15.37MB anonymously downloaded/SHA-verified;4 retained native sequences exact. Particle export and membership pending. |
| Public dataset collection | Policy corpus and3-archive training dataset published/verified. Final G.001 archive and benchmark outputs pending. |
| GitHub exact commands/source provenance | Clean public HyperGAN/evidence-dial export, pinned sources and historical owned files by SHA. Private research history/full FaithEval passages excluded. Final commands/results pending. |
| Public HF demo | Live ZeroGPU create actually refused402 at account cap; declared static fallback. DOM checks pass. Final benchmark/deploy pending; actual browser host unavailable, visual check unmeasured. |
| Article and under500-word abstract | README fallback and packet draft prepared; final metrics/abstract verification pending. |
| New versus pre-existing work and licence | Binding/corpus/pipeline are new hackathon work; Particle Sliders core pre-existing. Tiny Aya and derived adapters CC-BY-NC-4.0; own policy corpus CC0. |
| Account checks and final form | User confirmed Discord and will fill form. Registration HF username matching ntc-ai remains an owner credit check. No agent submission is claimed. |
| Releases notification | Authenticated Ember reports queued for policy/AlignTune/archive. Receipt and onward Releases delivery unverified; final release notice pending. |
