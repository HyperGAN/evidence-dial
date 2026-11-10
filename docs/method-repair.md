# Learning from Jev-selected Tiny Aya successes

The corrected experiment trains on model-produced solutions that passed a
recorded correctness check. It uses the shared Particle Sliders response API,
activation-conditioned routed bank, ParticleGAN0.9.5 game, RAW checkpoints and
native inference. The earlier answer-hinted experiment remains a separate
comparison; none of its results are relabelled as this method.

## Two distinct target recipes

The original variant constructed targets for all64 source cases by adding the
known gold answer to the teacher prompt. It omitted successful-sample curation.
The corrected variant first asks frozen Tiny Aya to explain its decision from
the evidence and provide an explicit final answer, without giving it the gold
answer, source labels or answer hints. A fixed maximum16 seeded attempts is
available for each source case. Every completed attempt is retained.

Code rejects incomplete outputs and final answers that fail the unchanged exact
alias check. Jev1.14, served through NanoGPT, then evaluates factual correctness
and explanation quality. Admission requires correctness≥0.90 and quality≥3 on
the0..4 rubric. The first qualifying attempt is retained. No threshold is
relaxed to fill a case; unsuccessful cases are listed explicitly.

Mining finished all64 source cases and retained48:16 answerable,16 altered-fact,
9 missing-evidence and7 conflict cases. The other16 remain inspectable. These
are training examples only; development and holdout do not supply targets.
The miner uses an explanation prompt; the trained student's neutral prompt is
exactly the unchanged short-answer evaluation prompt.

## What the seed and latent records mean

The generation cache saves the actual seed, initial CPU/CUDA RNG states, input
and output token IDs, sampling settings, model revision, generated text and the
native2048-value hidden state before every sampled token. A seed alone is not a
complete replay specification. Jev's exact questions, typed first response,
usage and content hash are cached separately.

The paired training cache follows the earlier Qwen workflow: both branches see
the same evidence and question. The positive branch additionally receives the
accepted Tiny Aya explanation/answer as private verified context. Both receive
the identical causal prefix of the accepted final answer. Their frozen native
final hidden vectors are saved together with exact input IDs and links to the
accepted generation, seed, sampled trajectory and Jev record. This experiment
contains118 such pairs. Each completed pair is reused rather than recomputed.

The sampled-generation trace and paired-target cache serve different purposes.
The trace proves where the successful example came from. The training target is
the frozen positive-context hidden state, not a claim that the two prompts have
identical internal activations. This retains the earlier Qwen pairing method;
its historical candidate generation was reference-assisted, while this retry's
candidate generation is unhinted.

The particle adapter itself routes each layer's current rank8 activation
features over a learned128×4 bank, then computes its correction from those
features and the routed latent. There is no independently drawn language-model
z. Within each ParticleGAN phase, real and residual-shifted fake share the same
sampled rows and critic noise. Critic noise is separate from native model state.
See [the objective details](training-objective.md) for the shared formulation.

## Declared retry and controls

The new backend is `tiny-aya-evidence-dial-curated-v1`. Its single fixed recipe
uses G0.001/D0.018/bank0.03, batch2, rank8, seed20261110 and128 updates, without
weight EMA or an automatic extension. Initialization applies only to the new
components and preserves pretrained weights and zero up-projections. New native
checks established OFF=initialON,2593 connected adapter tensors, exact FULL
restore and a working RAW native-only inference path.

The existing qualified native-text catastrophe check runs at2/32/64/128;
confirmed collapse stops the candidate. Weak effect is a separate measurement.
Development strengths1 then2 select by equal-case strict finished success ON
minus matched OFF, preserving ties. Teacher-response SSE and training loss do
not select a model.

A fresh AlignTune LoRA control uses exactly the same48 approved source cases,
the unchanged strong prompt and128 updates×batch2, with the original fixed
supervised recipe and RAW weights. Its targets are the canonical short answers.
The objectives and exposure units differ: particle training samples118 response
positions; SFT samples48 cases. Actual journals, preparation cost and exposure
counts are reported. Original64-case LoRA and all previous results remain.

Before retry training,768 new balanced confirmation cases from32 fresh families
were frozen. All four answer categories occur equally at document counts1/2/3,
with384 matched forward/reversed pairs. The principal new comparison is the
development-selected particle model against the same-pool LoRA; OFF and both
original adapters remain controls. Both nominal particle strengths are reported
without selecting a test winner. Earlier tests are repeated research comparisons,
not a new pristine confirmation set.

## Curation limitations and audit trail

Eight fixed Jev fixtures were judged once. The filter admitted2/4 valid examples
and0/4 invalid ones; two valid examples received0.89 and were rejected at the
unchanged0.90 threshold. The original full-recall qualification failure remains
recorded. A separate conservative-filter admission documents these false
negatives and does not turn this small check into a universal correctness claim
or a catastrophic-observer qualification. No first judgment was rerated.

One legacy diagnostic-only response-error phase omitted explicit CUDA mapping:
allocator residency was measured for50.98s, but GPU1 owned-NVML telemetry was0.
Its physical device is unverified. The original record and diagnostic values
remain; no effect score or selection used them. The direct diagnostic helper now
maps the requested physical UUID before model loading. The cost report separates
this exception from certified GPU1 phases; it cannot claim that every historical
phase met the one-device plan.

The immutable declarations and source audit are under
`artifacts/method-repair-v1`; actual curation/pair records are public in
[the successful-sample archive](https://huggingface.co/datasets/ntc-ai/evidence-dial-curated-successes).
Run records stay outside Git and all original/corrected checkpoints and journals
are public in the training archive. The complete corrected evaluation is below.

## Development result before fresh confirmation

The corrected run completed128 updates and passed its checkpoint and both
nominal-strength output checks. Its development results are negative:
strength1 scores39/128 and strength2 scores32/128, versus matched OFF86/128.
The same48-case AlignTune control scores128/128. Strength1 is the fixed
development selection because its equal-case effect is less negative; that
selection does not establish an improvement. Strength2 emits UNKNOWN for all
128 cases. Legal abstention passes the bounded output-failure instrument,
while the per-type answer metrics expose excessive abstention.

The supervised run completed and saved all128 updates in98.19s. Its subsequent
development phase stopped at8 committed answers when shared GPU free memory
fell below5GiB. The worker exited and released its training allocator. A named
inference-only recovery retained those8 outputs and generated only the120
missing ones; journal and weight hashes stayed unchanged. The first refusal
and execution receipts remain. No training or completed answer was replayed.

A separate CPU output-head audit uses only retained hidden vectors and the
frozen tied embedding/output weights. All118 corrected positive positions put
their intended next answer token first in float32 projection; all48 accepted
answer prefixes are supported at their cached positions. This is teacher-forced
training-state evidence, without an EOS check or new transformer forward. It
does not establish finished-answer accuracy, choose a model, or explain the
optimization failure by itself. Cache populations and answer tokenizations
differ between original and corrected recipes; cross-recipe token counts are
descriptive. See `artifacts/evidence/frozen-cache-output-head-audit-v1.json`.


## Complete matched results and publication

All declared comparisons are complete and independently verified: 17,868 native
case-arm measurements across 2,772 distinct cases. Selected strength 1 scores
501/900 on repeated FaithEval versus matched LoRA 270/900 (+25.67pp,
source-question 95% interval [22.69,28.67]). Its missing-evidence count is
278/300 versus 104/300. The fresh policy confirmation instead scores 209/768
versus LoRA 477/768 (−34.90pp, family interval [−38.41,−31.38]), with 678 literal
UNKNOWN responses. The original policy, cue and balanced comparisons also
regress. Conflict handling remains weak. Neither test result changes selection.

The strict FaithEval MC rule rejects a correct letter plus exact option text.
A separate post hoc diagnostic accepts that unambiguous format, preserving the
primary scorer. MC counts become 220/214/221 of 300 for new particle/matched
LoRA/original LoRA; their particle choice-difference intervals include zero.
Overall particle still wins 504/900 versus 325/900 and 416/900. The supported
transfer result is unsupported-answer refusal, not established semantic choice
superiority. FaithEval is repeated research, with no verified matched answerable
QA controls. The fresh primary-use-case failure is equally part of the result.

The public corrected particle and matched LoRA each reproduce four native
saved generations. One and only one saved candidate was replayed from its RNG
record: output tokens and hidden-state bits match exactly, maximum difference
zero. Native teacher-relative SSE at strengths 1/2 is 2.936/5.504, worse than
OFF; this remains diagnostic. The CPU head check above uses forced prefixes
without EOS and does not contradict the failed free-generation result.

The benchmark, model cards, all training archives and both-recipe explorer are
linked from the final README. Complete native input pairing and all primary
statistics were checked from retained outputs without rerunning evaluation.
After measurement, prospective CLI defaults were repaired to select the frozen
recipe-v2 and completed curation-v4; exact executed source versions remain
captured by SHA. Those default changes did not modify measured weights or data.
