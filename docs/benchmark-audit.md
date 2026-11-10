# Why the original perfect score needed better tests

Our own harness measured the AlignTune-trained LoRA at256/256 on the original
policy holdout. An independent CPU audit reproduced256 raw exact gold matches,
without aliases, normalization or AuditKIT scoring. Every generated token
sequence decodes to its recorded output; every native input equals the prompt
built from evidence/question only. Changing gold/type/ID metadata leaves that
prompt unchanged. Actual training journals contain only the64 training IDs.
Families and exact prompts are disjoint across splits.

That validates the narrow score, not the benchmark's breadth. All16 question
frames appear in every split. DocumentB perfectly predicts CONFLICT across all
448 original cases. The base already answers64/64 ordinary supported holdout
questions. Family names alone therefore give inadequate separation of general
reading behavior from template cues. The original results remain published:
OFF169/256, selected particle184/256, LoRA256/256.

Two new tests were frozen before any generation on them:

- **80-case intervention audit:** five fixed patterns across16 fresh families.
  Agreeing sources, wrong-scope second sources, contradiction within one source,
  missing facts across two sources, and agreeing nonstandard simulator facts.
  This is a diagnostic stress audit, not a replacement primary benchmark.
- **768-case balanced test:**32 new families ×four categories ×three document
  counts ×two orders. Every category occurs equally at every count. Questions,
  fields, values and rendering forms differ from training. Gold derives from
  the unique values matching the query's exact scope and field, then is checked
  against parsed rendered facts. All384 order pairs retain identical truth.

CuratorKIT schema/dedup checks passed all768. Maximum complete chat input646
tokens; no evidence truncation or post-output exclusions. Manual review fixed
simulator question grammar before model calls; the unused v1 draft and its
zero-call withdrawal are retained. Active balanced data is v2.

The original selection remains Particle G0.001/strength1. The larger fresh test
compares that fixed candidate with OFF and the saved LoRA. The80-case diagnostic
also retains nominal strength2. No training, selector or strict scorer changed.
All results, including failures and paired family-bootstrap intervals, will be
reported. The32 families are the independent bootstrap units;768 is the number
of conditions/orders, not768 independent facts. New formats remain synthetic
and shared within the test. The frozen900-case FaithEval evaluation supplies
external evidence.

The owner requested a comparative advantage over LoRA. The prospective overall
FaithEval comparison was declared before any FaithEval output, with all three
strata reported. Footprint is unfavorable: particle4,766,912 learned parameters
versus LoRA3,833,856. No size or speed advantage is claimed. The new tests assess
whether the apparent accuracy ceiling survives evidence changes, without
selecting favorable examples or changing the saved models.

Evidence: `artifacts/evidence/policy-ceiling-audit-v1.json`, dataset manifests,
`artifacts/evidence/secondary-comparison-plan-v1.json`, and source builders under
`tools/`. Public frozen tests:
https://huggingface.co/datasets/ntc-ai/evidence-dial-grounding-tests
