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
Production deployments typically mount all three directories from a persistent
volume.

Provide the four steering assets via the GitHub Releases bootstrap flow
(`SAE_BOOTSTRAP_MODEL=1`, see [README.md](../README.md)) or by placing them
manually under `server/plugins/steering/{models,data}/`:

- `models/TopKSAE-1024.ckpt` — SAE checkpoint
- `data/item_sae_features_TopKSAE-1024.pt` — runtime activations
- `data/llm_labels_TopKSAE-1024_llm.json` — neuron label cache
- `data/semantic_merged_TopKSAE-1024.json` — semantic cluster index

For Railway-hosted deployments see [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md).
