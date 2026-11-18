# Tiny Aya model and dataset releases

The model and dataset each have a release manifest in `releases/` and a separate
Hugging Face Trusted Publisher. A push to `main` runs the publication workflow
when a manifest or its release tooling changes. CI uses a short-lived token
scoped to the exact Hub repository. The initial manifests point to the existing
public revisions and let CI verify its permission and readback before the next
artifact release.

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
