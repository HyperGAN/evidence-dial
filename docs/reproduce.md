# Reproducibility

Published predictions are the authoritative first measurements. Recomputing
scores from them requires no GPU, API key, teacher cache or model generation.
Fresh training is a separate experiment; never overwrite or replay this cohort.

## Environment and source

The measured host used Python3.11.17, Torch2.14.1+cu130 and Transformers4.57.6.
The finite package versions are in `configs/training-environment.json`; kit
commits are in `configs/training-sources.lock.json`. The local environment
inherited host Torch. Installing a newer all-extras AlignTune bundle is not a
reproduction of the measured combination.

Start with a compatible CUDA PyTorch environment, then stage the pinned sources:

```bash
python3.11 -m venv --system-site-packages .venv
.venv/bin/python tools/bootstrap_sources.py
.venv/bin/python -m pip install --no-deps -e vendor/CuratorKIT -e vendor/AuditKIT -e vendor/particle-sliders/packages/particle-sliders-core -e .
.venv/bin/python -m pip install transformers==4.57.6 huggingface-hub==0.35.0 particlegan==0.9.5 peft==0.18.0 trl==0.25.1 datasets==4.4.1 accelerate==1.12.0 pyarrow==21.0.0 fsspec==2025.9.0 dill==0.4.0 multiprocess==0.70.18 safetensors==0.7.0 numpy==1.26.4
.venv/bin/python -m pytest -q
hf download CohereLabs/tiny-aya-global --revision af89d219b53ed9b13b8a4645f9c8028973510324
```

The base is gated: use your own approved HF account. No base weights or
credentials are included in this repository. CuratorKIT, AuditKIT and AlignTune
retain their upstream licences. The project code is MIT, our fictional corpus
is CC0, and derived model weights follow Tiny Aya's CC-BY-NC-4.0 licence.

## Frozen data

Public policy files are included in this checkout and in the policies dataset.
Use those frozen files directly; build-policy below is for an empty new cohort.
FaithEval contexts are fetched
from the original repositories at the revisions in `faith.py`; the public
release includes frozen IDs, order, exclusions and hashes, rather than
republishing their full passages or reversible input-token arrays.

```bash
# Reconstruct the already frozen FaithEval IDs; no gated tokenizer/model needed.
.venv/bin/python tools/materialize_faith.py --out artifacts/data/faith-materialized-v1
# Only for a new, empty corpus/cohort:
.venv/bin/evidence-dial build-policy --out artifacts/data/policy-rebuild-v1
.venv/bin/evidence-dial freeze-faith --out artifacts/data/faith-refreeze-v1
```

Materialization uses the published900 IDs/order and token-count metadata, and
asserts the exact original frozen digest before writing. Its full JSONL bytes
were checked against the retained original. It never samples again. The full
freeze command uses eligibility before seeded
sampling and never truncate evidence. Check generated manifests against the
published frozen manifests. Source-question IDs shared across FaithEval strata
were audited: all19 shared IDs map to the same bootstrap family.

## Jev-curated retry

The corrected method is a distinct experiment, described in
[method-repair.md](method-repair.md). Its frozen inputs are
`artifacts/method-repair-v1/recipe-v2.json`, the48 accepted cases in
`artifacts/curation-success-v4/selected.json`, and the118-position paired cache.
Do not substitute the original146-position gold-hinted cache below. The new
fresh confirmation corpus is
`artifacts/data/balanced-grounding-confirm-v1/holdout.jsonl`; its768 cases and
comparison plan were frozen before the retry committed an update.

The original mining pass made455 seeded native generations. Exact sampled
tokens, CPU/CUDA RNG states and hidden trajectories were retained, with65
successful Jev requests including8 fixtures. Use these retained artifacts for
exact reproduction. A fresh API judgment is a new observation, even with the
same prompt and model name. Jev's first full-recall fixture check failed because
two valid examples scored.89; its conservative-filter admission preserves
those first results and the unchanged.90/3.0 acceptance thresholds. That
admission cannot be copied to a different runtime without checking its scope.

The public archive restores the exact measured `artifacts/` paths. In an empty
checkout, download and verify it before extracting:

```bash
hf download ntc-ai/evidence-dial-curated-successes curation-and-pairs-v1.tar.gz --repo-type dataset --revision 1338cf30f4fa643ff3de295d5087f8572b678955 --local-dir .downloads/curated
sha256sum .downloads/curated/curation-and-pairs-v1.tar.gz
tar -xzf .downloads/curated/curation-and-pairs-v1.tar.gz -C .
```

Expected archive SHA256:
`c3d290f7887ec49a0173e1563c2f0a203cebc477c8f594bd9738171ae026d175`.
Extraction creates an already-completed paired cache: skip cache generation
and use its saved values. It contains no base weights or training checkpoint.

For a new, explicitly bounded mining experiment, provide your own NanoGPT key
through `NANOGPT_API_KEY` and use an empty output directory. The code refuses
ambiguous already-committed API calls. Run qualification first and inspect its
actual result before mining; never lower a threshold or rerate a first judgment
to make it pass.

```bash
.venv/bin/python -m evidence_dial.curation qualify --out artifacts/curation-new
.venv/bin/python -m evidence_dial.curation mine --out artifacts/curation-new --recipe artifacts/method-repair-v1/recipe-v2.json --attempt first
```

On the measured host the following phases consumed the retainedv4 curation.
Run them only with empty new cache/run paths, or reuse the already completed
cache/checkpoints. The source-bound run paths shown identify the measured
cohort; do not execute them over an existing release.

```bash
.venv/bin/python -m evidence_dial.curated_train cache --curation artifacts/curation-success-v4 --cache artifacts/cache/curated-success-v1 --run artifacts/runs/particle-curated-G001-v1 --attempt first
.venv/bin/python -m evidence_dial.curated_train native-check --cache artifacts/cache/curated-success-v1 --run artifacts/runs/particle-curated-G001-v1 --attempt first
.venv/bin/python -m evidence_dial.curated_train raw-check --cache artifacts/cache/curated-success-v1 --run artifacts/runs/particle-curated-G001-v1 --attempt first
.venv/bin/python -m evidence_dial.curated_train train --cache artifacts/cache/curated-success-v1 --run artifacts/runs/particle-curated-G001-v1 --stop 2 --attempt first
.venv/bin/python tools/observe_checkpoint.py --candidate artifacts/runs/particle-curated-G001-v1 --step 2
```

Continue the train/check pair at32,64,128 only after a qualified passing guard.
The actual run was stopped and reaped at committed FULL68 to enlarge its
execution window, then resumed exactly68→128. The68 checkpoint received its
own passing check before continuation. No completed update was replayed and
the fixed128-update horizon was unchanged.

The supervised control uses the identical48 accepted source-case IDs, their
original strict gold completions and the original AlignTune recipe with
`train_cases=48` and `dataloader_drop_last=true`. Its recipe is retained as
`artifacts/method-repair-v1/aligntune-curated-recipe-v1.json`. It commits128
updates at batch2. Particle training samples paired answer positions; LoRA
samples whole cases. The journals, rather than a claim of identical exposure,
establish what each optimizer saw.

The measured supervised call was the following, in a separate child process.
Use it only for a new empty run. If `training-result.json` already records128,
skip training and reuse the saved adapter/checkpoint.

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import os, subprocess
os.environ['CUDA_VISIBLE_DEVICES'] = subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
from evidence_dial.aligntune_train import train
train(Path('artifacts/data/policy-curated-success-v1'), Path('artifacts/runs/aligntune-curated-v1'), 1, recipe_path=Path('artifacts/method-repair-v1/aligntune-curated-recipe-v1.json'))
PY
```

That trainer also evaluates development after saving. In the measured retry,
its reserved training allocator left insufficient shared free memory after8
answers. All128 updates were already saved. The child was reaped and only
missing development outputs were recovered using this fresh-process pattern;
`evaluate_rows` preserves committed answers and reuses any retained capture.
Use a new resource-attempt name after an inspected terminal failure:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import os, subprocess
os.environ['CUDA_VISIBLE_DEVICES'] = subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
from peft import PeftModel
from evidence_dial.evaluate import load_base, tokenizer, NativeGenerator, evaluate_rows
from evidence_dial.resources import Resources
from evidence_dial.io import read_jsonl
run = Path('artifacts/runs/aligntune-curated-v1')
with Resources(1, run/'dev/resources-dev-only-recovery-v1.json') as resource:
    model = PeftModel.from_pretrained(load_base(), str(run/'adapter'), is_trainable=False).eval()
    evaluate_rows(read_jsonl(Path('artifacts/data/policy-curated-success-v1/dev.jsonl')), run/'dev', NativeGenerator(model, tokenizer()), arm='aligntune-lora', resource=resource)
PY
```

The retained recovery verified the unchanged journal and weight SHA256, kept
the original8 answer bytes, added120 missing cases, and passed its final guard.
It did not resume the training optimizer. No completed recovery needs rerunning.

Selection uses only the128 original development cases, strengths1 then2 and
equal-case finished-answer ON−OFF success. The retry writes a separate
`method-repair-v1/selection-v1.json` before any new holdout output. It cannot
replace the old selection. Fresh confirmation compares against the same-pool
LoRA; repeated earlier tests retain their status as research comparisons.

## Original answer-hinted training experiment

The following are the actual workflow commands, with the published recipe and
an empty output root. On an existing completed checkout, use its saved states
and outputs instead. Jobs declare one physical GPU, sequenced by a private
lease; allocator22GiB, own NVML24GiB, RSS32GiB, free GPU floor5GiB and host
available floor47GiB. Create `artifacts/execution-budget.json` with a new explicit
24-hour UTC deadline before a new experiment. The measured cohort's deadline
was November10 at23:41UTC; it is not extended by resuming a process. One old
diagnostic-only50.98s phase omitted the device mapping: GPU residency was
measured, but its physical device is unverified. Its receipt is preserved in
`artifacts/evidence/diagnostic-device-mapping-audit-v1.json`. The direct
diagnostic helper now maps the requested physical UUID before CUDA use.

```bash
.venv/bin/evidence-dial headroom --data artifacts/data/policy-v1 --out artifacts/runs/headroom-v3 --gpu 1
.venv/bin/evidence-dial train-lora --data artifacts/data/policy-v1 --out artifacts/runs/aligntune-v2 --gpu 1
.venv/bin/evidence-dial cache --data artifacts/data/policy-v1 --out artifacts/cache/teacher-v1 --headroom artifacts/runs/headroom-v3/gate.json --gpu 1
```

Only cache when the teacher beats OFF. Native qualification of the first fresh
arm uses `evidence_dial.train.native_check`; subsequent declared arms use the
same initial receipt and reuse that measured proof. The exact phase commands:

```bash
.venv/bin/python tools/run_phase.py native-check --run artifacts/runs/particle-G012-v1 --rate .012 --gpu 1
.venv/bin/python tools/run_phase.py train --run artifacts/runs/particle-G012-v1 --stop 2 --gpu 1
.venv/bin/python tools/observe_checkpoint.py --candidate artifacts/runs/particle-G012-v1 --step 2
.venv/bin/python tools/run_phase.py train --run artifacts/runs/particle-G012-v1 --stop 32 --gpu 1
.venv/bin/python tools/observe_checkpoint.py --candidate artifacts/runs/particle-G012-v1 --step 32
```

A confirmed catastrophe permanently stops that LR. G0.012 actually failed at32;
there is no RAW128 score for it. For each surviving rate, continue only after a
passing checkpoint guard, to64 then128. G0.004 and G0.001 are fresh declared
arms, never resets of G0.012:

```bash
.venv/bin/python tools/run_phase.py fresh --run artifacts/runs/particle-G004-v1 --rate .004 --reference artifacts/runs/particle-G012-v1 --gpu 1
.venv/bin/python tools/run_phase.py train --run artifacts/runs/particle-G004-v1 --stop 2 --gpu 1
.venv/bin/python tools/observe_checkpoint.py --candidate artifacts/runs/particle-G004-v1 --step 2
```

Repeat the explicit next-boundary train/check pair at32,64,128, and then use
`.001` with `particle-G001-v1`. Do not blindly retry an interrupted phase:
inspect/reap its actual process, retain its execution receipt and resume the
latest committed FULL, without replaying updates. A new bounded attempt gets a
new stable `--attempt` name. FULL saves precede update-journal commits.

## Qualified observer

The first v5 guard reused an existing Qwen3-VL8B image/text qualification.
Its actual decoding was sampled at temperature0.7/top-k20/top-p0.8. It then
rejected two outcomes explicitly allowed by the frozen text contract: a literal
UNKNOWN label and a normal64-token cutoff. The old raw verdicts, guard states
and qualified7-case proof remain intact. This is not a reason to erase a real
repetition failure or convert an uncertain case into a pass.

A distinct native-text instrument was frozen and qualified on8 new fixtures:
UNKNOWN,CONFLICT,A,19months, coherent wrong text, coherent token-limit truncation,
pervasive repetition and gibberish. All8 first observations matched their fixed
expectations. It explicitly uses greedy decoding and `use_model_defaults=False`;
this does not retroactively change the older sampled instrument. Its scope is
bounded English short answers/prose, not all media or languages. The Jev
alternative returnedHTTP402 before a semantic result; no retry or fabricated
probability followed.

The active wrapper stages model weights before committing a semantic observation,
so a loading-floor refusal stays unmeasured. Identical retained observations are
reused. Audits in `guard_scope.py` bind the exact old v5 failure/source/output
bytes and only the two proved contract contradictions. Original failed files
are immutable. An audit permits a replacement assessment, not training or
promotion: the exact checkpoint and disputed output group must pass the new
qualified check. Any new confirmed catastrophe remains permanently blocked.
G0.012's actual CONF repetition has no applicable scope audit.

`repro/observer-layout-native-text-v1e.zip` contains the finite source, contracts,
package/model manifests, actual8-case observations and qualification records.
It contains no model weights or original image fixture bytes. The exact
measured-host entrypoint `tools/observe_checkpoint.py` intentionally requires its
qualified interpreter/artifact layout. The bundle does not silently qualify a
new machine. Requalify a changed runtime explicitly before training/promotion;
never fabricate a pass or automatically rerate a committed first observation.
The G0.004 strength2 check subsequently flagged a truncated mathematical
fragment. Mathematical notation was not established by the8 fixtures; its
qualification is recorded as uncertain and the LR stays blocked. No qualifier
was retuned to admit it. Its complete nominal effects remain visible, with
numerical ties preserved separately from eligibility.

Portable adapter inference below is separate from strict source-bound FULL and
native-v2 training packets.

## Development, selection and holdout

```bash
.venv/bin/python tools/run_eval.py --attempt G004-dev-strength1-first -- dev-particle --artifact artifacts/runs/particle-G004-v1/NATIVE-000128.pt --data artifacts/data/policy-v1/dev.jsonl --out artifacts/runs/particle-G004-v1/dev-strength1 --strength 1 --gpu 1
.venv/bin/python tools/run_eval.py --attempt G004-dev-strength2-first -- dev-particle --artifact artifacts/runs/particle-G004-v1/NATIVE-000128.pt --data artifacts/data/policy-v1/dev.jsonl --out artifacts/runs/particle-G004-v1/dev-strength2 --strength 2 --gpu 1
```

Evaluate1 before2 for each completed rate. A fallback strength is allowed only
under the plan's declared conditions and must be written before its outputs
are generated. `benchmark.select` requires complete nominal arms or persistent
qualified failures, preserves effect ties and refuses an existing selection.
Freeze selection before calling `evidence-dial evaluate --selection ...` on
holdout. All strict scoring lives in the frozen `protocol.py`; teacher-response
SSE in `diagnose.py` is diagnostic only.

## Portable RAW inference

The portable safetensors export maps the exact committed RAW tensor values to
logical names; it contains no teacher, critic, optimizer or base weights and
cannot resume training. Use the selected published model ID or downloaded
portable directory with:

```bash
.venv/bin/evidence-dial infer --adapter ntc-ai/evidence-dial-particle --row demo/example.json --strength 1 --out example-output.json --gpu 1
```

The shared prompt, greedy native EOS3 and64-new-token cap are built in. Input
over2048 tokens is refused without truncation. Equivalence to retained original
native-v2 generation is reported separately; export validity alone is no
behavioral improvement claim.

Historical owned source versions are retained by content SHA in
`repro/source-history`, including the original teacher-cache source. Private
Git history, account facts and full FaithEval passages are excluded.

## Validate the saved-output explorer

The explorer has no inference dependency. Its DOM behavior can be checked with
Node24 and JSDOM26.1.0, without starting a browser or calling a model:

```bash
npm install --prefix .ui-test --no-audit --no-fund jsdom@26.1.0
node tools/check_demo.cjs
```

This checks the strength selector, matched response cards, dataset/type filters,
pager wrapping, empty states and escaping of model output text. It does not
establish rendered CSS layout or a real browser visual check.

`tools/build_demo.py` reads only a complete retained results release. FaithEval
passages are excluded; its explorer entries link to the pinned upstream source.
Policy evidence is our own fictional CC0 corpus. A complete benchmark's native
input sequences, budgets and EOS are checked privately by
`tools/audit_native_pairs.py`; the public receipt contains counts and hashes,
not reversible source input token IDs.

## Recompute a complete corrected release

These operations consume only saved outputs and make zero model/API calls.
The release builder refuses missing case-arm measurements; run it only after
the declared comparison index is complete. Use a new output directory:

```bash
.venv/bin/python -c "from pathlib import Path; from evidence_dial.summarize import build; build(Path('artifacts/method-repair-v1/release-index-v1.json'), Path('publication/corrected-benchmark-v1'))"
.venv/bin/python tools/verify_results.py --release publication/corrected-benchmark-v1 --policy artifacts/data/policy-v1/holdout.jsonl --faith artifacts/data/faith-v1/holdout.jsonl --additional-data confirmation=artifacts/data/balanced-grounding-confirm-v1/holdout.jsonl --additional-data cue_audit=artifacts/data/evidence-interventions-v1/holdout.jsonl --additional-data balanced=artifacts/data/balanced-grounding-v2/holdout.jsonl
.venv/bin/python tools/audit_native_pairs.py --index artifacts/method-repair-v1/release-index-v1.json --out artifacts/evidence/corrected-native-pair-audit-v1.json
.venv/bin/python tools/training_diagnostics.py --include-curated --out artifacts/evidence/corrected-training-v1
.venv/bin/python tools/cost_report.py --out artifacts/evidence/corrected-resource-cost-v1.json
.venv/bin/python tools/build_demo.py --release publication/corrected-benchmark-v1 --out demo/benchmark.json
node tools/check_demo.cjs
```

The public FaithEval metadata must first be materialized from the pinned
upstream sources as above; pass that exact byte-identical file with the
verifier's `--faith` option when it is outside the measured default path.
Resource receipts include mining, target preparation, training, inference and
observer phases where measured. The report marks legacy telemetry gaps and
does not interpret a provider's reported `estimated_cost=0` as verified billing.


## Download the complete public results

The final combined benchmark is pinned at
`bbebf36c42294ea0814a397147b5809f1e846b04`. Its default configuration contains
2,772 paired cases; the original release is preserved as a separate config.
All result recomputation is CPU-only and requires no fresh model/API call:

```bash
hf download ntc-ai/evidence-dial-benchmark --repo-type dataset --revision bbebf36c42294ea0814a397147b5809f1e846b04 --local-dir .downloads/benchmark
.venv/bin/python - <<'PYDATA'
from datasets import load_dataset
rows = load_dataset('ntc-ai/evidence-dial-benchmark', 'curated_paired_cases', revision='bbebf36c42294ea0814a397147b5809f1e846b04', split='test')
assert len(rows) == 2772
PYDATA
.venv/bin/python tools/build_demo.py --release .downloads/benchmark/curated-retry-v1 --out demo/benchmark.json
```

`curated-retry-v1/results.json` records all strict comparisons and intervals;
`paired-cases.jsonl` contains matched predictions; `predictions/` retains the
individual arm files. `diagnostics/faith-mc-labelled-option-diagnostic-v1.json`
is a post hoc formatting check, not a replacement primary score. The root
manifest provides byte counts and SHA256 values. Training archive revision
`310f70935d2b02a136ab6c0f7dfc2214347a5f48` preserves all six original/corrected
run archives, including all 129 new particle FULL states. The model cards link
the portable inference files and exact distribution proofs.

Current CLI defaults point to recipe-v2 and the completed curation-v4. These
prospective defaults were fixed after measurement; their executed earlier
sources are retained in source-history. No measurement, proof or completed
update was rerun to accommodate the defaults.
