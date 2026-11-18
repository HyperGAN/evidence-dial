# Tiny Aya demo releases

The existing [Tiny Aya Space](https://huggingface.co/spaces/ntc-ai/tiny-aya-particle-sliders)
publishes the static files named in [the allowlist](../tools/space-files.txt) from
`demo/`. Add new public app files to that list in the same PR. An owner handles
removing an old public Space file explicitly. Publish model and dataset artifacts
through their versioned release procedures.

The GitHub workflow runs the recorded-answer rendering contract and compares the
candidate with the Space. A push to `main` publishes only changed files,
conditioned on the Space revision observed before publication, then verifies the
repository files and three public routes. For identical payloads, the workflow
logs the verified live revision.

The release workflow uses the [Space Trusted Publisher](https://huggingface.co/spaces/ntc-ai/tiny-aya-particle-sliders/settings).
It exchanges the GitHub Actions identity for a short-lived token scoped to this
Space. The publisher matches this repository, the `main` push, the exact
workflow ref, and the `huggingface-space-production` environment.

The commit message on Hugging Face records the GitHub revision. For a rollback,
revert the affected demo change on GitHub and let the next `main` release restore
the prior file contents.
