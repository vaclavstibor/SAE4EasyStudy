# ELSA + TopKSAE Training (ml-32m-filtered)

## Scope
- Dataset: `ml-32m-filtered`
- CF model: `ELSA` only
- SAE model: `TopKSAE` only
- Default split: `80:20` (`train:val`, `test=0`)

## Run from
- Directory: `train/recsys26`

## 1) Train ELSA
```bash
export DATASET=ml-32m-filtered
bash run_cf_model_training.sh
```

What this does:
- Trains ELSA for embedding sizes `512, 1024, 2048`
- Uses early stopping
- Saves checkpoints and JSON metrics

## 2) Train TopKSAE on ELSA
```bash
export DATASET=ml-32m-filtered
bash run_topksae_cosine_model_training.sh
```

What this does:
- Loads only `ELSA-*` checkpoints from `checkpoints/ml-32m-filtered`
- Trains `TopKSAE` with cosine reconstruction over a small grid
- Tracks downstream quality retention vs base ELSA

## Outputs
- Checkpoints: `train/recsys26/checkpoints/ml-32m-filtered`
- Metrics JSON: `train/recsys26/results/ml-32m-filtered`
- Notebook loader example: `train/recsys26/example.ipynb`

## Key metrics to watch
- ELSA validation: `Recall@10`, `Recall@20`, `nDCG@10`, `nDCG@20`
- TopKSAE retention and sparsity:
  - `recall degradation`, `ndcg degradation`
  - `cosine`, `l0`, `dead neurons`
