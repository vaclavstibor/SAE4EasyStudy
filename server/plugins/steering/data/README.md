# Steering Data

This directory is owned by the `server/plugins/steering/` plugin.

Required runtime data artifacts for the current SAE study:

- `item_embeddings.pt`
- `item_sae_features_TopKSAE-1024.pt`
- `llm_labels_TopKSAE-1024_llm.json`
- `semantic_merged_TopKSAE-1024.json`

Two JSON files can be copied from the older local repository:

- `/Users/vaclav.stibor/Downloads/SAE4EasyStudyRecSys26-main/server/plugins/sae_steering/data/llm_labels_TopKSAE-1024_llm.json`
- `/Users/vaclav.stibor/Downloads/SAE4EasyStudyRecSys26-main/server/plugins/sae_steering/data/semantic_merged_TopKSAE-1024.json`

The checkpoint and `.pt` runtime tensors are not present in that older tree and
must come from the release/bootstrap source or from a previously prepared local
asset folder.
