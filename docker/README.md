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

For the SAE plugin, only the two JSON metadata files are available in the older
local repo copy:

- `/Users/vaclav.stibor/Downloads/SAE4EasyStudyRecSys26-main/server/plugins/sae_steering/data/llm_labels_TopKSAE-1024_llm.json`
- `/Users/vaclav.stibor/Downloads/SAE4EasyStudyRecSys26-main/server/plugins/sae_steering/data/semantic_merged_TopKSAE-1024.json`

The checkpoint and `.pt` tensor files must come from the release/bootstrap flow
or from an existing prepared asset directory.

## Local use

From repository root:

```bash
cp docker/compose.env.example .env
docker compose up --build
```
