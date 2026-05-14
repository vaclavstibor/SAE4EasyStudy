# Steering Data

This directory belongs to the `server/plugins/steering/` plugin. The application
expects four runtime data artifacts before the steering blueprint can serve
recommendations:

- `item_embeddings.pt`
- `item_sae_features_TopKSAE-1024.pt`
- `llm_labels_TopKSAE-1024_llm.json`
- `semantic_merged_TopKSAE-1024.json`

(The matching SAE checkpoint lives in
[`../models/TopKSAE-1024.ckpt`](../models/).)

## Supported flows

There are two supported ways to obtain these files:

1. **GitHub Releases bootstrap (recommended).** Set
   `SAE_BOOTSTRAP_MODEL=1` along with `SAE_MODEL_GITHUB_REPO` and
   `SAE_MODEL_RELEASE_TAG` (or use `latest`); the container entrypoint runs
   [`bootstrap_model.py`](../bootstrap_model.py) and downloads every asset into
   the correct location. The same script is also runnable directly:

   ```bash
   python -m server.plugins.steering.bootstrap_model --tag v1.0.0
   ```

2. **Manual placement.** Drop the four files into this directory and the
   checkpoint into `../models/` yourself. The container entrypoint validates
   their presence on startup and refuses to launch if any are missing.

The container entrypoint refuses to start with `DATASET_BOOTSTRAP=1`; the
MovieLens-derived dataset under `server/static/datasets/ml-32m-filtered/` must
be placed manually in all environments.
