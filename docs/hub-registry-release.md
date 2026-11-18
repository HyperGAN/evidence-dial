# Tiny Aya and Evidence Dial model and dataset releases

Each active repository has a release manifest in `releases/` and a separate
Hugging Face Trusted Publisher. A push to `main` runs the publication workflow
when a manifest or its release tooling changes. CI uses a short-lived token
scoped to the exact Hub repository. The initial manifests point to the existing
public revisions. CI exercises the Hub upload preflight with an empty file list
and reads back the production revisions before the next artifact release.

The Evidence Dial model and dataset repositories keep their original weights,
corpora, test cases, and numerical evidence at their released paths. Stage
corrections to a model or dataset card in `README.md`. Add new versioned files
for reviewed artifacts, and include any Hugging Face LFS metadata changes in
the same manifest.

The original Policies dataset is sourced from `artifacts/data/policy-v1`.
The public `policy/` tree at `581d29da86c022f2ad3835171513f362b9ab62bd`
matches all seven versioned source files byte for byte.

1. Build and validate the model or dataset with its existing artifact tooling.
   Record the evaluation or dataset quality evidence in the GitHub PR.
2. Stage the approved files on a branch of the corresponding Hub repository
   created from its current `main` revision. Use a new versioned path for
   historical artifacts. A branch in a public Hub repository is public.
3. Use the full staged commit SHA when generating the release manifest from this
   checkout. Run with `huggingface_hub==1.19.0` installed.

   ```sh
   python tools/publish_hub_release.py model --stage <staged-sha> > releases/huggingface-model.json
   ```

   For the dataset, use `dataset` and `releases/huggingface-dataset.json`.
   For a historical Evidence Dial publication, pass the exact repository ID.

   ```sh
   python tools/publish_hub_release.py dataset --repo ntc-ai/evidence-dial-benchmark \
     --stage <staged-sha> > releases/huggingface-evidence-dial-benchmark.json
   ```
4. Review the manifest delta together with the staged files and merge the PR.
   The workflow rechecks file identities and the base revision, copies only
   the reviewed changes, and reads back the complete production tree. The Hub
   commit records the GitHub source and staged artifact revisions.

If another publication has advanced Hub `main`, rebuild the manifest against the
new base and review it again. An exact retry of a successful release reports its
current Hub revision. Consumers can pin the earlier `base_revision` while a
corrected release is prepared.

Publisher settings use the GitHub Actions issuer with exact claims for the
canonical repository and numeric repository ID, `refs/heads/main`, the
`huggingface-registry.yml` workflow ref, `huggingface-registry-production`
environment, `push` event and `github-hosted` runner. The publisher grants only
`repo.content.write` to its model or dataset destination.
