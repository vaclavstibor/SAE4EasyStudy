# Deployment

This directory contains the minimal runtime configuration surface for non-compose
deployments.

## Canonical deployment model

The current deployment story is intentionally simple:

- one application image
- one explicit DB bootstrap step (`server/scripts/init_db.py`)
- one runtime entrypoint (`server.platform.app:create_app()`)

In Docker-based deployments this is already wrapped by
`server/docker-entrypoint.sh`.

## Files

- `app.env.example` lists the core environment variables expected by the app
  runtime and gunicorn process.

## Expected mounted assets

Production or staging deployments still need these assets available:

- dataset directory: `server/static/datasets/ml-32m-filtered/`
- steering models: `server/plugins/steering/models/`
- steering data files: `server/plugins/steering/data/`

If those are not baked into the image, mount them before starting the app.

Only these two steering JSON files are available in the older local repo copy:

- `/Users/vaclav.stibor/Downloads/SAE4EasyStudyRecSys26-main/study_framework/server/plugins/sae_steering/data/llm_labels_TopKSAE-1024_llm.json`
- `/Users/vaclav.stibor/Downloads/SAE4EasyStudyRecSys26-main/study_framework/server/plugins/sae_steering/data/semantic_merged_TopKSAE-1024.json`

The checkpoint and `.pt` runtime tensors must be provided from the release or an
already prepared local asset folder.
