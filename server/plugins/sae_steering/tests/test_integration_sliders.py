#!/usr/bin/env python
"""Integration test: runs _personalized_features on real SAE model,
checks diversity and slider sensitivity on actual recommendations.

Run:  cd server && .venv39/bin/python plugins/sae_steering/tests/test_integration_sliders.py
"""
import sys, os, json, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from plugins.sae_steering import (
    _personalized_features,
    _expand_feature_adjustments,
    _jaccard_similarity,
    _text_token_set,
)
from plugins.sae_steering.sae_recommender import get_sae_recommender

SEED = [1006, 721, 1035, 969, 564, 921, 236]
MODEL = "TopKSAE-1024"
NUM_SLIDERS = 14


def main():
    features = _personalized_features(selected_movies=SEED, model_id=MODEL, num_sliders=NUM_SLIDERS)

    print("=" * 70)
    print("SELECTED %d SLIDERS for seed %s" % (len(features), SEED))
    print("=" * 70)

    for i, f in enumerate(features):
        ms = f.get("_member_scores", {})
        mids = f.get("member_ids", [f["id"]])
        top_scores = dict(list(ms.items())[:3])
        print("  %d. %s (N%s) | movies=%s | group=%d members | scores=%s" % (
            i + 1, f["label"], f["id"], f.get("movie_count", 0), len(mids), top_scores))

    # -- Diversity check --
    print("\nPAIRWISE LABEL JACCARD (should all be < 0.50):")
    violations = 0
    for i, a in enumerate(features):
        for j, b in enumerate(features):
            if i >= j:
                continue
            t_a = _text_token_set(a["label"])
            t_b = _text_token_set(b["label"])
            jac = _jaccard_similarity(t_a, t_b)
            if jac >= 0.30:
                tag = "FAIL" if jac >= 0.50 else "warn"
                print("  [%s] vs [%s] Jaccard=%.2f %s" % (a["label"], b["label"], jac, tag))
                if jac >= 0.50:
                    violations += 1
    print("  Violations (>= 0.50): %d" % violations)

    # -- Slider sensitivity --
    print("\nSLIDER SENSITIVITY TEST (delta=+1.0 and -1.0):")

    rec = get_sae_recommender(model_id=MODEL)
    rec.load()

    id_to_idx = {}
    for i, mid in enumerate(rec.item_ids):
        id_to_idx[int(mid)] = i

    acts_list = []
    for m in SEED:
        idx = id_to_idx.get(int(m))
        if idx is not None:
            acts_list.append(rec.item_features[idx].cpu().numpy())
    mean_act = np.mean(acts_list, axis=0)

    seed_adj = {}
    for nid in range(len(mean_act)):
        if mean_act[nid] > 0:
            seed_adj[str(nid)] = round(float(mean_act[nid]), 4)

    baseline_recs = rec.get_recommendations(
        feature_adjustments={int(k): float(v) for k, v in seed_adj.items()},
        n_items=50, exclude_items=SEED,
        seed_adjustments={int(k): float(v) for k, v in seed_adj.items()},
    )
    baseline_ids = [r["movie_id"] for r in baseline_recs[:20]]

    sensitive_count = 0
    total_tests = 0
    for f in features:
        for delta in [1.0, -1.0]:
            total_tests += 1
            expanded = _expand_feature_adjustments(
                raw_adjustments={str(f["id"]): delta},
                current_features=features,
                cluster_map={},
            )
            combined = dict(seed_adj)
            for k, v in expanded.items():
                combined[k] = combined.get(k, 0) + v

            steered_recs = rec.get_recommendations(
                feature_adjustments={int(k): float(v) for k, v in combined.items()},
                n_items=50, exclude_items=SEED,
                seed_adjustments={int(k): float(v) for k, v in seed_adj.items()},
            )
            steered_ids = [r["movie_id"] for r in steered_recs[:20]]

            changed = sum(1 for i in range(min(20, len(steered_ids)))
                          if i >= len(baseline_ids) or steered_ids[i] != baseline_ids[i])
            overlap = len(set(baseline_ids[:10]) & set(steered_ids[:10]))
            ok = changed >= 4
            tag = "OK" if ok else "WEAK"
            if ok:
                sensitive_count += 1

            sign = "+1.0" if delta > 0 else "-1.0"
            label = f["label"] if delta > 0 else ""
            print("  %-40s %s: changed=%2d/20 top10_overlap=%d %s" % (label, sign, changed, overlap, tag))

    pct = 100 * sensitive_count / total_tests if total_tests else 0
    print("\nSensitive: %d/%d (%.0f%%)" % (sensitive_count, total_tests, pct))
    print("Diversity violations: %d" % violations)

    if violations > 0:
        print("\nFAIL: diversity violations")
        return 1
    if pct < 50:
        print("\nFAIL: sensitivity below 50%%")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
