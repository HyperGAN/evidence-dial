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

Public policy files are in the policies dataset. FaithEval contexts are fetched
from the original repositories at the revisions in `faith.py`; the public
release includes frozen IDs, order, exclusions and hashes, rather than
republishing their full passages or reversible input-token arrays.

```bash
.venv/bin/evidence-dial build-policy --out artifacts/data/policy-v1
.venv/bin/evidence-dial freeze-faith --out artifacts/data/faith-v1
```

These commands refuse an existing cohort. They use eligibility before seeded
sampling and never truncate evidence. Check generated manifests against the
published frozen manifests. Source-question IDs shared across FaithEval strata
were audited: all19 shared IDs map to the same bootstrap family.

## New training experiment on the measured host

The following are the actual workflow commands, with the published recipe and
an empty output root. On an existing completed checkout, use its saved states
and outputs instead. All jobs use one physical GPU, sequenced by a private
lease; allocator22GiB, own NVML24GiB, RSS32GiB, free GPU floor5GiB and host
available floor47GiB. Create `artifacts/execution-budget.json` with a new explicit
24-hour UTC deadline before a new experiment. The measured cohort's deadline
was November10 at23:41UTC; it is not extended by resuming a process.

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
