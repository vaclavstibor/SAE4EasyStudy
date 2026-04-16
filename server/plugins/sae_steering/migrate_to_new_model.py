#!/usr/bin/env python3
"""
One-time migration: set up TopKSAE-1024 model artifacts for the sae_steering plugin.

Run on the machine where checkpoints are available (e.g. Azure VM):

    cd EasyStudy
    python server/plugins/sae_steering/migrate_to_new_model.py \
        --sae-ckpt  train/recsys26/checkpoints/ml-32m-filtered/TopKSAE-1024-4d51a427.ckpt \
        --elsa-ckpt train/recsys26/checkpoints/ml-32m-filtered/ELSA-512-c2005bb7.ckpt \
        --dataset-dir data_preparation/filters/recsys26/ml-32m-filtered \
        --labeling-run labeling/artifacts/20260416-023922

Produces inside server/plugins/sae_steering/:
    models/TopKSAE-1024.ckpt
    data/item_embeddings.pt
    data/llm_labels_TopKSAE-1024_llm.json
    data/semantic_merged_TopKSAE-1024.json
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PLUGIN_DIR = Path(__file__).resolve().parent
NEW_MODEL_ID = "TopKSAE-1024"


def _load_sorted_item_ids(dataset_dir: Path):
    ratings = dataset_dir / "ratings.csv"
    if not ratings.exists():
        raise FileNotFoundError(f"Missing {ratings}")
    item_values = []
    with open(ratings, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        col = "movieId" if "movieId" in reader.fieldnames else "item_id"
        for row in reader:
            item_values.append(str(row[col]))
    # Match train/recsys26 ordering exactly (np.unique over strings).
    # NOTE: this is lexicographic, not integer sort.
    ordered = np.unique(np.array(item_values, dtype=str))
    return [int(v) for v in ordered.tolist() if str(v).isdigit()]


def _extract_elsa_embeddings(elsa_ckpt: Path, item_ids):
    payload = torch.load(str(elsa_ckpt), map_location="cpu", weights_only=False)
    encoder = payload["model_state_dict"]["encoder"].detach()
    embeddings = F.normalize(encoder, dim=1)
    if embeddings.shape[0] != len(item_ids):
        print(
            f"  WARNING: ELSA encoder rows ({embeddings.shape[0]}) != item_ids ({len(item_ids)})"
        )
    return embeddings


def _find_labeling_artifact(run_dir: Path, pattern: str):
    """Find an artifact file, tolerating the -N version suffix."""
    candidates = sorted(run_dir.glob(pattern))
    if not candidates:
        candidates = sorted(run_dir.glob(pattern.replace(".json", "*.json")))
    if not candidates:
        return None
    return candidates[-1]


def main():
    ap = argparse.ArgumentParser(description="Migrate sae_steering plugin to TopKSAE-1024")
    ap.add_argument("--sae-ckpt", type=Path, required=True)
    ap.add_argument("--elsa-ckpt", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--labeling-run", type=Path, required=True)
    args = ap.parse_args()

    models_dir = PLUGIN_DIR / "models"
    data_dir = PLUGIN_DIR / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy SAE checkpoint
    dst_ckpt = models_dir / f"{NEW_MODEL_ID}.ckpt"
    print(f"[1/4] Copying SAE checkpoint -> {dst_ckpt}")
    shutil.copy2(str(args.sae_ckpt), str(dst_ckpt))
    print(f"  done ({dst_ckpt.stat().st_size / 1e6:.1f} MB)")

    # 2. Generate item_embeddings.pt from ELSA
    print(f"[2/4] Generating item_embeddings.pt from ELSA checkpoint")
    item_ids = _load_sorted_item_ids(args.dataset_dir)
    print(f"  item_ids: {len(item_ids)}")
    embeddings = _extract_elsa_embeddings(args.elsa_ckpt, item_ids)
    print(f"  embeddings shape: {tuple(embeddings.shape)}")
    dst_emb = data_dir / "item_embeddings.pt"
    torch.save({"embeddings": embeddings, "item_ids": item_ids}, str(dst_emb))
    print(f"  saved -> {dst_emb}")

    # 3. Copy labeling artifacts (llm_labels, semantic_merged)
    print(f"[3/5] Copying labeling artifacts")
    run_dir = args.labeling_run

    llm_src = _find_labeling_artifact(run_dir, "llm_labels_*_llm*.json")
    if llm_src:
        dst = data_dir / f"llm_labels_{NEW_MODEL_ID}_llm.json"
        shutil.copy2(str(llm_src), str(dst))
        print(f"  {llm_src.name} -> {dst.name}")
    else:
        print("  WARNING: no llm_labels artifact found")

    merged_src = _find_labeling_artifact(run_dir, "semantic_merged_*.json")
    if merged_src:
        dst = data_dir / f"semantic_merged_{NEW_MODEL_ID}.json"
        shutil.copy2(str(merged_src), str(dst))
        print(f"  {merged_src.name} -> {dst.name}")
    else:
        print("  WARNING: no semantic_merged artifact found")

    # 4. Clean up legacy artifacts
    print(f"[4/5] Cleaning legacy artifacts")
    for legacy in [
        data_dir / f"cluster_profile_{NEW_MODEL_ID}.json",
    ]:
        if legacy.exists():
            legacy.unlink()
            print(f"  removed {legacy.name}")

    # 5. Summary
    print(f"\n[5/5] Migration complete. Plugin model_id: {NEW_MODEL_ID}")
    print(f"  models/ : {dst_ckpt.name}")
    print(f"  data/   : item_embeddings.pt")
    if llm_src:
        print(f"  data/   : llm_labels_{NEW_MODEL_ID}_llm.json")
    if merged_src:
        print(f"  data/   : semantic_merged_{NEW_MODEL_ID}.json")
    print(f"\n  item_sae_features_{NEW_MODEL_ID}.pt will auto-compute on first server start.")


if __name__ == "__main__":
    main()
