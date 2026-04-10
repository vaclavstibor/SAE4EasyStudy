#!/usr/bin/env python3
"""
Pre-generate LLM-based neuron labels for a SAE model.

This is a MANDATORY offline step before deploying the server. LLM labels
are the only labeling path — there is no TF-IDF fallback at runtime.

Usage:
    # With a local llama.cpp server running on port 8080:
    export SAE_LLM_BACKEND=openai
    export SAE_LLM_API_BASE=http://localhost:8080/v1
    export SAE_LLM_MODEL=llama-3-8b-instruct
    python generate_llm_labels.py --model prediction_aware_sae

    # With llama-cpp-python (no server needed):
    export SAE_LLM_BACKEND=llamacpp
    export SAE_LLM_GGUF_PATH=/path/to/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
    python generate_llm_labels.py --model prediction_aware_sae

    # Force recompute (ignore cache):
    python generate_llm_labels.py --model prediction_aware_sae --force

The script will:
  1. Load the SAE model and compute item features.
  2. For each active neuron, collect top-N activating movies.
  3. Prompt the LLM to generate a structured label + description (JSON).
  4. Cache results to data/llm_labels_<model_id>_llm_v2.json
"""

import argparse
import os
import sys
import time

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def main():
    parser = argparse.ArgumentParser(description="Generate LLM-based neuron labels")
    parser.add_argument("--model", default="prediction_aware_sae",
                        help="SAE model ID (e.g. prediction_aware_sae)")
    parser.add_argument("--force", action="store_true",
                        help="Force recompute, ignore cache")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test LLM connection and label 3 neurons only")
    args = parser.parse_args()

    print(f"=== LLM Neuron Labeling (v2 — structured labels) ===")
    print(f"Model: {args.model}")
    print(f"Backend: {os.environ.get('SAE_LLM_BACKEND', 'openai')}")
    print(f"API Base: {os.environ.get('SAE_LLM_API_BASE', 'http://localhost:8080/v1')}")
    print(f"LLM Model: {os.environ.get('SAE_LLM_MODEL', 'llama-3-8b-instruct')}")
    print()

    # Load SAE recommender
    from sae_recommender import get_sae_recommender
    recommender = get_sae_recommender(model_id=args.model)
    recommender.load()

    if recommender.item_features is None:
        print("ERROR: SAE model has no item_features. Cannot generate labels.")
        sys.exit(1)

    print(f"Loaded model: {args.model}")
    print(f"  item_features shape: {recommender.item_features.shape}")
    print(f"  item_ids count: {len(recommender.item_ids)}")
    print()

    # Test LLM connection
    from llm_labeling import _get_llm, get_llm_labels
    llm = _get_llm()
    print("Testing LLM connection...")
    if llm.is_available():
        print("  ✓ LLM is reachable")
    else:
        print("  ✗ LLM is NOT reachable. Check your configuration:")
        print(f"    SAE_LLM_BACKEND={os.environ.get('SAE_LLM_BACKEND', 'openai')}")
        print(f"    SAE_LLM_API_BASE={os.environ.get('SAE_LLM_API_BASE', 'http://localhost:8080/v1')}")
        print(f"    SAE_LLM_MODEL={os.environ.get('SAE_LLM_MODEL', 'llama-3-8b-instruct')}")
        sys.exit(1)

    if args.dry_run:
        print("\n--- DRY RUN: labeling first 3 active neurons ---")
        import numpy as np
        import torch
        features_np = recommender.item_features.cpu().numpy() if isinstance(
            recommender.item_features, torch.Tensor) else recommender.item_features
        active_neurons = []
        for nid in range(features_np.shape[1]):
            if np.sum(features_np[:, nid] > 0) >= 20:
                active_neurons.append(nid)
            if len(active_neurons) >= 3:
                break

        from llm_labeling import label_neurons_by_ids_llm
        labels = label_neurons_by_ids_llm(
            active_neurons, args.model,
            recommender.item_features, recommender.item_ids
        )
        for nid, info in labels.items():
            print(f"\n  N{nid}:")
            print(f"    Label:       \"{info['label']}\"")
            print(f"    Description: \"{info.get('description', 'N/A')}\"")
            print(f"    Source:      {info.get('label_source', '?')}")
            print(f"    Genres:      {info.get('genres', [])}")
            print(f"    Tags:        {info.get('tags', [])}")
        print("\nDry run complete. Use without --dry-run to label all neurons.")
        return

    # Full run
    # offline_mode=True: bypass cache early-return, enter labeling loop with resume.
    # --force: also clears existing cache to start from scratch.
    if args.force:
        import glob
        cache_pattern = os.path.join(os.path.dirname(__file__), 'data', f'llm_labels_{args.model}_*.json')
        for f in glob.glob(cache_pattern):
            os.remove(f)
            print(f"  Deleted: {f}")
    start = time.time()
    labels = get_llm_labels(
        model_id=args.model,
        item_features=recommender.item_features,
        item_ids=recommender.item_ids,
        force_recompute=args.force,
        offline_mode=True,
    )
    elapsed = time.time() - start

    # Summary
    llm_count = sum(1 for v in labels.values() if v.get('label_source') == 'llm')
    metadata_count = sum(1 for v in labels.values() if v.get('label_source') == 'metadata')
    print(f"\n=== Summary ===")
    print(f"Total neurons labeled: {len(labels)}")
    print(f"  LLM labels: {llm_count}")
    print(f"  Metadata fallback: {metadata_count}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"\nSample labels:")
    for nid, info in list(labels.items())[:10]:
        print(f"  N{nid:5d}: \"{info['label']}\" | {info.get('description', '')} "
              f"[{info.get('label_source', '?')}] "
              f"(act={info['activation_count']}, sel={info['selectivity']:.3f})")


if __name__ == "__main__":
    main()
