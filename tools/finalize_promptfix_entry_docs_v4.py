"""Resolve public artifact links and publish a focused setup-diagnosis note."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/behavior-promptfix-20261114-v4'
DATA = 'https://huggingface.co/datasets/ntc-ai/evidence-dial-behavior/blob/main/promptfix37-h1024-v4'
PUBLIC = 'https://ntc-ai-evidence-dial.static.hf.space/'


def pin(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    audit_path = BASE / 'locked-fresh64-v4/independent-completion-audit-first.json'
    audit = json.loads(audit_path.read_text())
    assert audit['total_test_answers'] == 384 and audit['all_answers_native_EOS']
    review = ROOT / 'publication/promptfix-source-review-v4'
    main = ROOT / 'README.md'
    generated = review / 'README-generated-v4-first.md'
    assert not generated.exists()
    generated.write_bytes(main.read_bytes())
    body = main.read_text()
    assert body.count(PUBLIC + '?case=') == 3
    body = body.replace(PUBLIC + '?case=', PUBLIC + 'promptfix-confirmation.html?case=')
    body = body.replace('README.md#separate-capability-check-against-sft)', 'README.md)')
    main.write_text(body)

    diagnosis = ROOT / 'docs/prompt-setup-diagnosis-v3.md'
    prior = review / 'diagnosis-before-final-first.md'
    assert not prior.exists()
    prior.write_bytes(diagnosis.read_bytes())
    diagnosis.write_text(f'''# Prompt setup diagnosis and repaired-target result

The earlier expanded 138-context study and its negative confirmation remain
public. This follow-up repairs the target construction, completes a new
development search, and evaluates one preselected adapter on 64 fresh questions.

## What was wrong with the setup

Every original target combined the behavioral goal with “Reply naturally and
concisely, in two or three short sentences.” That was unrelated to the intended
sympathetic, constructive response with independent judgment. On the old fresh
32, OFF averaged 387 tokens and the restricted prompt 64.

Actual saved tokenizer inputs show that Aya retains its default preamble and
appends the custom instruction. There was no default-preamble replacement bug.
All 138 accepted old training answers ended at EOS despite their 128-token
mining cap; accepted target cutoffs were not the cause.

## Prospective prompt comparison on 16 DEV questions

Four new conditions were frozen before generation. Both controls reused their
completed original full-answer outputs. New generations used the same pinned
base, greedy decoding, a 4,096-token cap and required native EOS. No completed
response or successful judge record was rerolled.

| Condition | Mean behavior score | Mean tokens |
| --- | ---: | ---: |
| OFF, reused | 0.662031 | 401.8 |
| Original behavior plus short sentences, reused | 0.666563 | 60.4 |
| Short-sentence instruction only | 0.604063 | 56.1 |
| Original behavior, length restriction removed | 0.705156 | 326.5 |
| Full goal, unrestricted | 0.700156 | 314.8 |
| Concrete support instructions, unrestricted | 0.703281 | 276.4 |

![Prompt comparison and answer lengths](figures/prompt-setup-DEV-ablation-v3.png)

Removing the restriction gains 0.043125 against OFF, with 12 wins and four
losses; its individual paired 95% interval is [0.018281, 0.069063]. Against the
old restricted prompt, the gain is 0.038594 with interval
[−0.011094, 0.089066]. These are descriptive DEV intervals after comparing
prompts. The experiment supports removing the unrelated constraint, but does
not isolate it as the sole cause of the earlier learned adapter's failure.
[Independent ablation audit]({DATA}/audits/prompt-setup-ablation.json).

## New curation and complete training

For every original TRAIN question, the unmodified base produced one complete
OFF answer and one answer under each of the three unrestricted instructions.
All 256 candidates reached EOS. The unchanged filter requires normalized effect
at least 0.75 and greater than matched OFF. One highest-scoring context per
accepted question yields 37 pairs, with coverage 8 discouragement / 12 unsupported
conclusion / 8 reasonable boundary / 9 uncertainty.

Twenty-six questions overlap the previous cache; eleven newly qualify and
eleven old accepted questions are excluded. Teacher-prompt counts are concrete
20 / full-goal 12 / unrestricted original 5. Selected TRAIN targets average
0.815338 versus matched OFF 0.706149. These selection-biased curation statistics
are not evaluation results. [Cache admission]({DATA}/audits/training-cache-admission.json).

Neutral and positive retain the same question and empty continuation history;
only positive receives the extra instruction. The original initialization,
The recorded ParticleGAN formulation and optimizer laws are retained. Weight EMA is
disabled. Two fresh cohorts complete 1,024 updates with batch 8, G0.00025 or
G0.0000625, D0.018 and bank0.03. Every quarter is independently admitted before
selection; all 384 DEV answers reach EOS.

![All 24 completed DEV settings](figures/promptfix-DEV-training-progress-24cells-v5.png)

Exact decimal selection finds one maximum: lower-rate step 1,024, strength 1,
mean 0.713906 versus OFF 0.662031 and prompt-only 0.705156. Both actual outer
controllers exit zero and all 13 children in each run are reaped. Final
strength-1 results improve each cohort's earlier maximum; intermediate and
other-strength results are mixed.
[Complete development closure]({DATA}/audits/development-search-closure.json).

## Fresh confirmation

The 64 synthetic questions were frozen before new PG training. They are exact
distinct messages from the previous 144, with broad scenario-family overlap.
The exact six conditions were locked before any confirmation inference: OFF,
corrected prompt-only, selected PG at strength 1, descriptive strengths 0.25
and 0.5, and the earlier expanded-study adapter reference.

Both actual GPU roots exit zero. All 384 full native answers, token inputs and
outputs, original typed Jev records, raw artifact identities, resource contexts
and no-EMA configuration pass independent audit. Selected PG scores 0.732656,
versus OFF 0.664180, prompt-only 0.697578 and earlier reference 0.668086.
The paired gain over OFF is +0.068477 [0.051211, 0.085392]; over prompting it
is +0.035078 [0.013358, 0.058320]. Intervals are individual question-bootstrap
intervals; there is no test-based reselection.
[All fresh comparison results]({DATA}/audits/fresh-confirmation-comparison.json).

Sympathy judgment rises to 0.802500 from OFF 0.687813. Sycophancy judgment is
0.088594 versus OFF 0.086094; it does not demonstrate a reduction. These are
separate diagnostics, never selectors. The main README includes full improvement
and regression examples; role consistency remains imperfect.

## Evaluation and inference repairs

Offline CPU replay found that AuditKIT's native run fingerprint omits
`experiment_name`. The same arm label could therefore reuse a result from a
different output directory. The qualified v5 wrapper includes the resolved
output namespace in CallableModel identity and records unmeasured exceptions.
Six saved-answer replays exercise the actual cache/generator/AuditKIT paths,
reuse four original Jev judgments, and reject unfinished or unknown committed
calls. The scorer, native decoder and Jev request remain unchanged. No new
model, API or confirmation calls occur in that qualification. Failed earlier
qualification attempts remain in the provenance archive.
[Namespace qualification]({DATA}/audits/confirmation-cache-namespace-v5-qualification.json).

The public `open_behavior_inference` owner calls the same 4,096-token/EOS
generator, validates the raw packet and exact strength, and loads once per
context. It makes no Jev calls. Its CPU contract preserves a real 544-token
DEV capture and rejects wrong-strength reuse, unfinished output, changed SHA,
EMA and a closed owner. The CPU fixture replaces the model-loading boundary;
it is not fresh-machine GPU replication. Native inference is separately
covered by actual experiment runs and reload proofs.
[Owner qualification]({DATA}/audits/fullanswer-owner-qualification.json).

## Scope and retained evidence

This is a small synthetic English study, one training seed and a shared
curation/evaluation judge. No human preference study, matched behavior-SFT
baseline or SOTA result is claimed. Prompt construction, accepted questions,
weighting and learning rate changed together. First-state supervision does
not directly evaluate private reasoning. A separate locally prepared role-aware
46-question / 138-context derivative remains untrained and contributes no result.

The [provenance archive]({DATA}/provenance/observations-and-sources.tar.gz)
contains original captures, raw judgments, plans, execution records, source
bodies and preserved failed checks with per-file hashes. Earlier full-answer
and short-answer studies remain archived. Raw/full checkpoints, actual update
counts, and the original optimizer formula are preserved.
''')
    observation = BASE / 'final-entry-docs-public-links-first.json'
    assert not observation.exists()
    observation.write_text(json.dumps({'state': 'final-entry-links-and-diagnosis-reviewed',
        'source': pin(__file__), 'audit': pin(audit_path), 'README': pin(main),
        'README_generated_before_link_review': pin(generated), 'diagnosis': pin(diagnosis),
        'diagnosis_prior_working_copy': pin(prior), 'new_model_or_Jev_calls': 0,
        'published': False}, indent=2) + '\n')
    print(json.dumps({'README': pin(main), 'diagnosis': pin(diagnosis),
        'observation': pin(observation), 'new_model_or_Jev_calls': 0}))


if __name__ == '__main__':
    main()
