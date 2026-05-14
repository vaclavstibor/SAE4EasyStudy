# Docker Assets

This directory documents the canonical local container runtime.

## Canonical files

- `../Dockerfile` builds the single application image.
- `../docker-compose.yml` starts the local stack.
- `../server/docker-entrypoint.sh` validates required assets, initializes the DB,
  and starts gunicorn.
- `compose.env.example` provides the supported compose-time overrides.

## Important path conventions

- SQLite volume mount: `/app/server/instance`
- cache volume mount: `/app/server/cache`
- dataset assets: `/app/server/static/datasets/ml-32m-filtered`
- steering model/data assets: `/app/server/plugins/steering/models` and
  `/app/server/plugins/steering/data`

The cache mount must point to `/app/server/cache`, because the current cache path
resolution is rooted under the `server/` package.

The steering plugin expects the following assets under
`server/plugins/steering/`:

- `models/TopKSAE-1024.ckpt` — SAE checkpoint
- `data/item_sae_features_TopKSAE-1024.pt` — runtime activations
- `data/llm_labels_TopKSAE-1024_llm.json` — neuron label cache
- `data/semantic_merged_TopKSAE-1024.json` — semantic cluster index

Provide them either through the GitHub Releases bootstrap flow
(`SAE_BOOTSTRAP_MODEL=1`, see [README.md](../README.md)) or by placing the
files manually under those paths before starting the container.

## Local use

From repository root:

```bash
cp docker/compose.env.example .env
docker compose up --build
```
