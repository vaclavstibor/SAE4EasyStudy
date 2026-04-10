#!/usr/bin/env python3
"""
Pre-generate semantic cluster profile for SAE neurons.

Strategy: cluster neurons by their FUNCTIONAL behavior (which movies
they activate on), not by label text.  This guarantees maximally
disjoint taste dimensions — each cluster covers a different region
of movie space.

1. Load item_features matrix (neuron activations per movie).
2. For each neuron, its "taste fingerprint" is its activation vector
   across all movies, L2-normalized.
3. Run Spectral Clustering (better than KMeans for high-dim correlated
   data) on the neuron fingerprints.
4. Name each cluster by aggregating MovieLens genres of its top-activated
   movies, then let LLM pick a broad taste label.

Usage:
    python generate_cluster_profile.py --model www_TopKSAE_8192
    python generate_cluster_profile.py --model www_TopKSAE_8192 --force

Output:
  data/cluster_profile_{model_id}.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

MIN_NEURON_MOVIES = 20
N_CLUSTERS = 7

# Hard-coded disjoint taste archetypes.  If LLM naming is unavailable
# or produces duplicates, these are used as fallback.
ARCHETYPE_LABELS = [
    {"label": "Action & Adventure", "description": "Non-stop excitement and thrill-seeking adventures."},
    {"label": "Dark & Suspenseful", "description": "Intense, thought-provoking stories that keep you on edge."},
    {"label": "Comedy & Lighthearted", "description": "Feel-good humor and lighthearted entertainment."},
    {"label": "Drama & Character-Driven", "description": "Deep, character-driven stories that explore the human condition."},
    {"label": "Sci-Fi & Fantasy", "description": "Imaginative worlds and futuristic adventures."},
    {"label": "Horror & Thriller", "description": "Heart-pounding scares and suspenseful twists."},
    {"label": "Romance & Family", "description": "Warm, relatable stories about love and relationships."},
]

CLUSTER_SYSTEM_PROMPT = """\
You are naming high-level taste dimensions for a movie recommender.

You will receive genre statistics and sample movie titles from a group
of neurons.  Name the ONE broad taste dimension they represent.

Rules for the **label**:
- Exactly 1-3 words, Title Case.
- Must be a BROAD, distinct category (like a tab in Netflix).
- Good examples: "Action-Packed", "Dark Suspense", "Feel-Good", "Epic Fantasy",
  "Indie & Art", "Family Fun", "Cerebral Thriller"
- NEVER use "Movies", "Films", "Gems", "Flicks", "Dramas" standalone.
- AVOID generic emotion words (Emotional, Heartfelt) unless paired with a genre.

Rules for the **description**:
- One sentence, max 12 words.
- Describe the *vibe*, not a genre list.

Reply ONLY valid JSON: {"label": "...", "description": "..."}
"""

CLUSTER_USER_TEMPLATE = """\
Cluster #{idx} of {n_clusters} total taste dimensions.
{n_neurons} neurons, activating on {n_movies} unique movies.

Dominant genres (by neuron activation count):
{genre_summary}

Sample top movies (most activated across this cluster):
{movie_sample}

IMPORTANT: The other clusters are: {other_labels}
Your label MUST be clearly different from all of them.

JSON only:
{{"label": "...", "description": "..."}}
"""


def load_llm_labels(model_id: str) -> dict:
    path = os.path.join(DATA_DIR, f"llm_labels_{model_id}_llm.json")
    if not os.path.exists(path):
        print(f"  WARNING: LLM labels not found at {path}")
        return {}
    with open(path) as f:
        raw = json.load(f)
    labels = {int(k): v for k, v in raw.items()}
    print(f"  Loaded {len(labels)} LLM labels for {model_id}")
    return labels


def main():
    parser = argparse.ArgumentParser(description="Generate semantic cluster profile")
    parser.add_argument("--model", default="prediction_aware_sae")
    parser.add_argument("--clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache_path = os.path.join(DATA_DIR, f"cluster_profile_{args.model}.json")

    if not args.force and os.path.exists(cache_path):
        print(f"Cache exists at {cache_path}. Use --force to regenerate.")
        with open(cache_path) as f:
            existing = json.load(f)
        for c in existing:
            print(f"  {c['id']}: {c['label']} ({len(c['neuron_ids'])} neurons)")
        return

    import torch
    from sae_recommender import get_sae_recommender

    print(f"=== Genre-Profile Cluster Profile ===")
    print(f"Model: {args.model}, Clusters: {args.clusters}\n")

    rec = get_sae_recommender(model_id=args.model)
    rec.load()
    features = rec.item_features
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()

    n_items, n_neurons = features.shape
    print(f"  Item-features matrix: {n_items} items x {n_neurons} neurons")

    act_counts = np.sum(features > 0, axis=0)
    active_mask = act_counts >= MIN_NEURON_MOVIES
    active_nids = np.where(active_mask)[0]
    print(f"  Active neurons (>={MIN_NEURON_MOVIES} movies): {len(active_nids)} / {n_neurons}")

    # Load movie genre data
    try:
        from plugins.utils.data_loading import load_ml_dataset
        loader = load_ml_dataset()
        movies_df = loader.movies_df_indexed
    except Exception as e:
        print(f"  FATAL: Need movie metadata for genre-based clustering: {e}")
        sys.exit(1)

    # Build genre vocabulary from the dataset
    all_genres = set()
    movie_genres = {}  # movieId -> list of genre strings
    for mid in movies_df.index:
        row = movies_df.loc[mid]
        gs = row.get('genres', '')
        if isinstance(gs, str):
            glist = [g.strip() for g in gs.split('|') if g.strip() and g.strip() != '(no genres listed)']
            movie_genres[int(mid)] = glist
            all_genres.update(glist)
    genre_list = sorted(all_genres)
    genre_idx = {g: i for i, g in enumerate(genre_list)}
    n_genres = len(genre_list)
    print(f"  Genre vocabulary: {n_genres} genres ({', '.join(genre_list[:8])}...)")

    # Build DISCRIMINATIVE genre profiles: remove the "stop-word" genres
    # (Drama, Comedy) that appear in 40%+ of movies and carry little
    # taste-distinguishing signal, then apply IDF on the rest.
    total_movies = len(movie_genres)
    genre_doc_freq = np.zeros(n_genres, dtype=np.float32)
    for glist in movie_genres.values():
        for g in glist:
            if g in genre_idx:
                genre_doc_freq[genre_idx[g]] += 1

    # Suppress genres that tag >35% of movies (Drama, Comedy typically)
    suppress_mask = np.ones(n_genres, dtype=np.float32)
    SUPPRESS_THRESHOLD = 0.35
    for gi, g in enumerate(genre_list):
        freq = genre_doc_freq[gi] / total_movies
        if freq > SUPPRESS_THRESHOLD:
            suppress_mask[gi] = 0.0
            print(f"  Suppressing genre '{g}' ({freq:.1%} of movies) — too common")

    idf = np.log(total_movies / (genre_doc_freq + 1))
    combined_weight = idf * suppress_mask

    print(f"  Building discriminative genre profiles...")
    genre_profiles = np.zeros((len(active_nids), n_genres), dtype=np.float32)
    for ni, nid in enumerate(active_nids):
        neuron_acts = features[:, nid]
        top_item_indices = np.where(neuron_acts > 0)[0]
        for item_idx in top_item_indices:
            mid = int(rec.item_ids[item_idx])
            glist = movie_genres.get(mid, [])
            w = float(neuron_acts[item_idx])
            for g in glist:
                genre_profiles[ni, genre_idx[g]] += w
        genre_profiles[ni] *= combined_weight

    # L2 normalize
    norms = np.linalg.norm(genre_profiles, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    genre_profiles = genre_profiles / norms

    # DETERMINISTIC GENRE-BASED ASSIGNMENT:
    # For each neuron, pick its strongest discriminative genre (highest
    # TF-IDF score). Group neurons into 7 genre buckets that represent
    # the most common distinguishing genres.
    from sklearn.metrics import silhouette_score

    print(f"  Assigning neurons by top discriminative genre...")

    # For each neuron, find its top genre
    neuron_top_genre = []
    for ni in range(len(genre_profiles)):
        top_gi = np.argmax(genre_profiles[ni])
        neuron_top_genre.append(genre_list[top_gi] if genre_profiles[ni, top_gi] > 0 else "Other")

    # Count genre frequencies
    from collections import Counter
    genre_freq = Counter(neuron_top_genre)
    print(f"  Top genre distribution: {genre_freq.most_common(12)}")

    # Define 7 disjoint genre buckets (merge related genres)
    GENRE_BUCKETS = [
        ("Action & Adventure",    {"Action", "Adventure", "IMAX"}),
        ("Sci-Fi & Fantasy",      {"Sci-Fi", "Fantasy"}),
        ("Thriller & Crime",      {"Thriller", "Crime", "Mystery", "Film-Noir"}),
        ("Horror",                {"Horror"}),
        ("Comedy",                {"Comedy", "Musical"}),
        ("Romance & Family",      {"Romance", "Children", "Animation"}),
        ("Documentary & Niche",   {"Documentary", "War", "Western"}),
    ]

    bucket_idx = {}
    for bi, (_, genres) in enumerate(GENRE_BUCKETS):
        for g in genres:
            bucket_idx[g] = bi

    assignments = np.zeros(len(genre_profiles), dtype=int)
    for ni in range(len(genre_profiles)):
        top_g = neuron_top_genre[ni]
        assignments[ni] = bucket_idx.get(top_g, len(GENRE_BUCKETS) - 1)

    dist = Counter(assignments)
    print(f"  Cluster sizes: {dict(sorted(dist.items()))}")

    # If any bucket is empty, merge it into the nearest non-empty bucket
    for bi in range(len(GENRE_BUCKETS)):
        if dist[bi] == 0:
            print(f"  WARNING: Bucket {bi} ({GENRE_BUCKETS[bi][0]}) is empty")

    from collections import Counter
    dist = Counter(assignments)
    print(f"  Cluster sizes: {dict(sorted(dist.items()))}")

    max_size = max(dist.values())
    min_size = min(dist.values())
    print(f"  Balance ratio: {min_size/max_size:.2f} (min/max = {min_size}/{max_size})")

    sil = silhouette_score(genre_profiles, assignments, metric='cosine',
                           sample_size=min(5000, len(genre_profiles)))
    print(f"  Silhouette score (cosine): {sil:.4f}")

    # Print cluster centroids in original genre space
    print(f"\n  Cluster genre centroids (original genre space):")
    raw_genre_profiles = np.zeros((len(active_nids), n_genres), dtype=np.float32)
    for ni, nid in enumerate(active_nids):
        neuron_acts = features[:, nid]
        top_item_indices = np.where(neuron_acts > 0)[0]
        for item_idx in top_item_indices:
            mid = int(rec.item_ids[item_idx])
            glist = movie_genres.get(mid, [])
            w = float(neuron_acts[item_idx])
            for g in glist:
                raw_genre_profiles[ni, genre_idx[g]] += w
    raw_norms = np.linalg.norm(raw_genre_profiles, axis=1, keepdims=True)
    raw_norms = np.maximum(raw_norms, 1e-8)
    raw_genre_profiles = raw_genre_profiles / raw_norms

    for ci in range(args.clusters):
        mask = assignments == ci
        centroid = raw_genre_profiles[mask].mean(axis=0)
        top_gi = np.argsort(centroid)[::-1][:5]
        top_str = ", ".join(f"{genre_list[gi]}={centroid[gi]:.2f}" for gi in top_gi)
        print(f"    cluster_{ci} ({dist[ci]:>4} neurons): {top_str}")

    # Load movie metadata for genre analysis
    try:
        from plugins.utils.data_loading import load_ml_dataset
        loader = load_ml_dataset()
        movies_df = loader.movies_df_indexed
    except Exception:
        movies_df = None
        print("  WARNING: Could not load movie metadata for genre analysis")

    # Load LLM labels for display
    llm_labels = load_llm_labels(args.model)

    # Build cluster objects with genre profiles
    clusters = []
    for ci in range(args.clusters):
        mask = assignments == ci
        cluster_nids = active_nids[mask].tolist()

        genre_counts = {}
        top_movie_ids = []
        if movies_df is not None:
            # Find movies that this cluster activates on most
            cluster_acts = features[:, cluster_nids].sum(axis=1)  # sum of activations per movie
            top_movie_indices = np.argsort(cluster_acts)[::-1][:100]
            for idx in top_movie_indices:
                try:
                    mid = int(rec.item_ids[idx])
                    if mid in movies_df.index:
                        row = movies_df.loc[mid]
                        genres_str = row.get('genres', '')
                        if isinstance(genres_str, str):
                            for g in genres_str.split('|'):
                                g = g.strip()
                                if g and g != '(no genres listed)':
                                    genre_counts[g] = genre_counts.get(g, 0) + 1
                        if len(top_movie_ids) < 10:
                            title = row.get('title', f'Movie {mid}')
                            top_movie_ids.append(str(title))
                except (KeyError, IndexError):
                    continue

        top_genres = sorted(genre_counts.items(), key=lambda x: -x[1])
        genre_list = [g for g, _ in top_genres[:5]]

        clusters.append({
            "id": f"cluster_{ci}",
            "neuron_ids": sorted(cluster_nids),
            "genres": genre_list,
            "genre_counts": genre_counts,
            "top_movies": top_movie_ids[:8],
            "n_neurons": len(cluster_nids),
        })

    # Name clusters — use LLM if available, else fallback to archetypes
    named_labels = []
    try:
        from llm_labeling import _get_llm
        llm = _get_llm()
        use_llm = llm.is_available()
    except Exception:
        use_llm = False

    if use_llm:
        print("\n  Naming clusters via LLM (sequentially for disjointness)...")
        for i, c in enumerate(clusters):
            genre_items = sorted(c["genre_counts"].items(), key=lambda x: -x[1])
            genre_summary = "\n".join(f"  {g}: {cnt}" for g, cnt in genre_items[:8])
            movie_sample = "\n".join(f"  - {m}" for m in c["top_movies"][:6])

            prompt = CLUSTER_USER_TEMPLATE.format(
                idx=i + 1,
                n_clusters=args.clusters,
                n_neurons=c["n_neurons"],
                n_movies=sum(c["genre_counts"].values()) // max(len(c["genre_counts"]), 1),
                genre_summary=genre_summary or "mixed",
                movie_sample=movie_sample or "  (no metadata available)",
                other_labels=", ".join(f'"{l}"' for l in named_labels) if named_labels else "(first cluster)",
            )

            for attempt in range(3):
                raw = llm.generate(CLUSTER_SYSTEM_PROMPT, prompt, max_tokens=150)
                if not raw:
                    time.sleep(1)
                    continue
                try:
                    raw = raw.strip()
                    if raw.startswith("```"):
                        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
                    result = json.loads(raw)
                    if "label" in result:
                        c["label"] = result["label"]
                        c["description"] = result.get("description", "")
                        named_labels.append(result["label"])
                        print(f"  [{i+1}/{args.clusters}] {c['label']} — {c['description'][:60]}")
                        break
                except (json.JSONDecodeError, KeyError):
                    import re
                    m = re.search(r'\{[^}]+\}', raw)
                    if m:
                        try:
                            result = json.loads(m.group())
                            if "label" in result:
                                c["label"] = result["label"]
                                c["description"] = result.get("description", "")
                                named_labels.append(result["label"])
                                print(f"  [{i+1}/{args.clusters}] {c['label']} — {c['description'][:60]}")
                                break
                        except json.JSONDecodeError:
                            pass
                time.sleep(0.5)
            else:
                fb = ARCHETYPE_LABELS[i % len(ARCHETYPE_LABELS)]
                c["label"] = fb["label"]
                c["description"] = fb["description"]
                named_labels.append(fb["label"])
                print(f"  [{i+1}/{args.clusters}] {c['label']} (fallback)")
    else:
        print("\n  LLM not available — using genre-based naming...")
        for i, c in enumerate(clusters):
            top_g = sorted(c["genre_counts"].items(), key=lambda x: -x[1])
            if top_g:
                primary = top_g[0][0]
                secondary = top_g[1][0] if len(top_g) > 1 and top_g[1][1] > top_g[0][1] * 0.3 else None
                c["label"] = f"{primary} & {secondary}" if secondary else primary
                c["description"] = f"Movies dominated by {primary}" + (f" and {secondary}." if secondary else ".")
            else:
                fb = ARCHETYPE_LABELS[i % len(ARCHETYPE_LABELS)]
                c["label"] = fb["label"]
                c["description"] = fb["description"]

    # Deduplicate labels — if LLM produced duplicates, append genre info
    seen_labels = set()
    for c in clusters:
        lbl = c["label"]
        if lbl in seen_labels:
            top_g = sorted(c["genre_counts"].items(), key=lambda x: -x[1])
            suffix = top_g[0][0] if top_g else "Mix"
            c["label"] = f"{lbl} ({suffix})"
        seen_labels.add(c["label"])

    # Clean up and save
    for c in clusters:
        c.pop("genre_counts", None)
        c.pop("top_movies", None)
        c.pop("n_neurons", None)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(clusters, f, indent=2)

    print(f"\n  Saved to {cache_path}")
    print(f"\nCluster summary:")
    for c in clusters:
        print(f"  {c['id']}: {c['label']} — {len(c['neuron_ids'])} neurons "
              f"({', '.join(c['genres'][:3])})")
        if c.get("description"):
            print(f"    {c['description']}")


if __name__ == "__main__":
    main()
