# Reviewed GitHub releases

Stage the approved files in a GitHub draft release. Set its target to the full
source commit used to prepare the release and use a tag that does not already
point to a different revision. The source commit must be in `main` history.
Include one narrated MP4. The workflow checks and decodes its video and audio.

Open a PR adding or updating `release.json` in the repository root.

| Field | Value |
| --- | --- |
| `tag` | New video version, such as `lexsi-video-v8` |
| `release_id` | Numeric ID of the draft GitHub release |
| `source_commit` | Full Git commit used to produce the reviewed release |
| `prerelease` | Boolean matching the draft's prerelease setting |
| `files` | Map of exact attachment filenames to `bytes` and `sha256` records |
| `media` | Reviewed `width`, `height` and `duration_ms` as positive integers |

Run `python tools/publish_github_release.py audit` with GitHub CLI access to the draft.
Keep the reviewed draft fixed while the request is open. Merging the request
rechecks the GitHub metadata, downloads and inspects each attachment, publishes
the draft, and verifies the public tag and attachment records. An exact retry
accepts the existing release after verifying it again.
