"""
SAE Neuron Analysis and Labeling via Activation-Weighted TF-IDF

Labels each neuron by computing which metadata attributes (genres, keywords,
actors, directors, decades) are most *discriminative* for that neuron relative
to all others. Uses TF-IDF: term-frequency within the neuron's top-k items,
inverse document-frequency across all neurons.

This avoids heuristic cascades and produces labels grounded in the same
methodology used by Anthropic's Scaling Monosemanticity (2024) -- examining
maximally-activating dataset examples and selecting the most specific
description.

Output:
    data/neuron_analysis.json  - full per-neuron analysis with exemplars
    data/neuron_labels.json    - neuron_id -> label mapping for the UI

Usage:
    python analyze_neurons.py [--prediction-aware]
"""

import json
import math
import pickle
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm
import re

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR.parent.parent / "static" / "datasets" / "ml-latest"

TOP_N_ITEMS_BASE = 30
TOP_N_ITEMS_MAX = 200


def load_data(use_prediction_aware=False):
    """Load ELSA embeddings, SAE model, and movie metadata."""
    print("Loading data...")

    with open(DATA_DIR / "item2index.pkl", "rb") as f:
        item2index = pickle.load(f)
    index2item = {v: k for k, v in item2index.items()}

    from train_elsa import ELSA, latent_dim
    elsa = ELSA(len(item2index), latent_dim)
    elsa.load_state_dict(torch.load(MODELS_DIR / "elsa_model_best.pt", map_location="cpu"))
    elsa.eval()
    elsa_embeddings = F.normalize(elsa.A.detach(), dim=1)

    if use_prediction_aware:
        from train_prediction_aware_sae import PredictionAwareSAE
        checkpoint = torch.load(MODELS_DIR / "prediction_aware_sae.pt", map_location="cpu")
        config = checkpoint["config"]
        sae = PredictionAwareSAE(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            k=config["k"],
            tied=config.get("tied", True),
        )
        sae.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        from train_sae import TopKSAE, hidden_dim, k
        sae = TopKSAE(latent_dim, hidden_dim, k)
        sae.load_state_dict(torch.load(MODELS_DIR / "sae_model_r4_k32.pt", map_location="cpu"))

    sae.eval()

    tmdb_path = DATASET_DIR / "tmdb_data.json"
    tmdb_data = {}
    if tmdb_path.exists():
        with open(tmdb_path, "r", encoding="utf-8") as f:
            tmdb_data = json.load(f)

    movies_path = DATASET_DIR / "movies.csv"
    movies = {}
    with open(movies_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                movie_id = parts[0]
                if len(parts) > 3:
                    title = ",".join(parts[1:-1]).strip('"')
                    genres = parts[-1]
                else:
                    title = parts[1].strip('"')
                    genres = parts[2]
                movies[movie_id] = {"title": title, "genres": genres.split("|")}

    return elsa_embeddings, sae, item2index, index2item, tmdb_data, movies


def extract_item_attributes(movie_id, tmdb_data, movies):
    """Extract typed metadata attributes for a single item."""
    tmdb_info = tmdb_data.get(str(movie_id), {})
    ml_info = movies.get(str(movie_id), {})
    title = tmdb_info.get("title") or ml_info.get("title", f"Movie {movie_id}")

    attrs = []  # list of (category, value) tuples

    genres = tmdb_info.get("genres") or ml_info.get("genres", [])
    if isinstance(genres, str):
        genres = genres.split("|")
    for g in genres:
        if g and g != "(no genres listed)":
            attrs.append(("genre", g))

    cast = tmdb_info.get("cast", [])
    for actor in cast[:5]:
        name = actor.get("name") if isinstance(actor, dict) else actor
        if name:
            attrs.append(("actor", name))

    for d in tmdb_info.get("directors", []):
        if d:
            attrs.append(("director", d))

    for kw in tmdb_info.get("keywords", []):
        if kw:
            attrs.append(("keyword", kw))

    year_match = re.search(r"\((\d{4})\)", title)
    if year_match:
        decade = f"{(int(year_match.group(1)) // 10) * 10}s"
        attrs.append(("decade", decade))
    elif tmdb_info.get("release_date"):
        try:
            decade = f"{(int(tmdb_info['release_date'][:4]) // 10) * 10}s"
            attrs.append(("decade", decade))
        except (ValueError, IndexError):
            pass

    return title, attrs


def analyze_all_neurons(use_prediction_aware=False):
    """
    Two-pass analysis:
      Pass 1 - collect per-neuron attribute counters and top items.
      Pass 2 - compute IDF across neurons, then TF-IDF label per neuron.
    """
    elsa_embeddings, sae, item2index, index2item, tmdb_data, movies = load_data(
        use_prediction_aware
    )

    print("Computing SAE activations...")
    with torch.no_grad():
        if use_prediction_aware:
            activations = sae.encode(elsa_embeddings)
        else:
            _, activations, _ = sae(elsa_embeddings)

    num_neurons = activations.shape[1]
    num_items = activations.shape[0]

    activation_counts = (activations > 0).sum(dim=0)
    selective_mask = (activation_counts > 0) & (activation_counts < num_items * 0.5)
    selective_neurons = selective_mask.nonzero().squeeze(-1).tolist()
    general_neurons = ((activation_counts >= num_items * 0.5)).nonzero().squeeze(-1).tolist()
    dead_neurons = (activation_counts == 0).nonzero().squeeze(-1).tolist()

    print(f"Selective: {len(selective_neurons)}, General: {len(general_neurons)}, Dead: {len(dead_neurons)}")

    # Pass 1: per-neuron attribute collection
    # attr_key = "category:value", neuron_attrs[neuron_idx] = Counter of attr_key -> count
    neuron_attrs: dict[int, Counter] = {}
    neuron_meta: dict[int, dict] = {}

    # Track which neurons contain each attribute (for IDF)
    attr_neuron_sets: dict[str, set] = defaultdict(set)

    all_neurons = selective_neurons + general_neurons
    for neuron_idx in tqdm(all_neurons, desc="Pass 1: collecting attributes"):
        neuron_acts = activations[:, neuron_idx]
        n_active = (neuron_acts > 0).sum().item()
        if n_active == 0:
            continue

        # Scale sample size with activation breadth: broader neurons need more
        # samples for keywords to cross the support threshold.
        top_n = min(max(TOP_N_ITEMS_BASE, n_active // 20), TOP_N_ITEMS_MAX, n_active)
        top_vals, top_indices = torch.topk(neuron_acts, top_n)

        attr_counter = Counter()
        top_movies = []
        genre_counter = Counter()
        cast_counter = Counter()
        director_counter = Counter()
        keyword_counter = Counter()
        decade_counter = Counter()

        for rank, (val, idx) in enumerate(zip(top_vals, top_indices)):
            idx_int = idx.item()
            act_val = val.item()
            if act_val <= 0:
                break

            movie_id = index2item[idx_int]
            title, attrs = extract_item_attributes(movie_id, tmdb_data, movies)

            top_movies.append({"id": str(movie_id), "title": title, "activation": round(act_val, 4)})

            for cat, val_str in attrs:
                key = f"{cat}:{val_str}"
                attr_counter[key] += 1
                attr_neuron_sets[key].add(neuron_idx)
                if cat == "genre":
                    genre_counter[val_str] += 1
                elif cat == "actor":
                    cast_counter[val_str] += 1
                elif cat == "director":
                    director_counter[val_str] += 1
                elif cat == "keyword":
                    keyword_counter[val_str] += 1
                elif cat == "decade":
                    decade_counter[val_str] += 1

        neuron_attrs[neuron_idx] = attr_counter
        neuron_meta[neuron_idx] = {
            "selective": neuron_idx in selective_neurons,
            "activation_sum": neuron_acts.sum().item(),
            "activation_count": n_active,
            "selectivity": round(1.0 - n_active / num_items, 4),
            "top_movies": top_movies[:5],
            "genres": dict(genre_counter.most_common(5)),
            "directors": dict(director_counter.most_common(3)),
            "cast": dict(cast_counter.most_common(3)),
            "keywords": dict(keyword_counter.most_common(5)),
            "decades": dict(decade_counter.most_common(3)),
        }

    # Neurons activating on very few items are degenerate (memorizing items,
    # not encoding concepts). Mark them but don't give them real labels.
    MIN_ACTIVATION_COUNT = 15

    # An attribute must appear in at least this many of a neuron's top items
    # to be considered for labeling. Prevents random actor names from winning.
    MIN_ATTR_SUPPORT = 3

    # When TF-IDF scores are close, prefer semantic categories over entities.
    # Higher bonus = stronger preference.
    CATEGORY_BONUS = {
        "keyword": 1.3,
        "genre": 1.2,
        "decade": 1.1,
        "director": 1.0,
        "actor": 0.8,
    }

    # Pass 2: compute IDF and generate TF-IDF labels
    total_neurons_analyzed = len(neuron_attrs)
    attr_idf = {}
    for attr_key, neuron_set in attr_neuron_sets.items():
        df = len(neuron_set)
        attr_idf[attr_key] = math.log(total_neurons_analyzed / (1 + df))

    neuron_info = {}
    neuron_labels = {}
    n_degenerate = 0

    for neuron_idx in tqdm(all_neurons, desc="Pass 2: TF-IDF labeling"):
        if neuron_idx not in neuron_attrs:
            continue

        meta = neuron_meta[neuron_idx]
        n_active = meta["activation_count"]

        if not meta["selective"]:
            meta["label"] = "General preference"
            meta["tfidf_scores"] = {}
            neuron_info[neuron_idx] = meta
            neuron_labels[neuron_idx] = meta["label"]
            continue

        if n_active < MIN_ACTIVATION_COUNT:
            n_degenerate += 1
            exemplar = meta["top_movies"][0]["title"] if meta["top_movies"] else "?"
            meta["label"] = f"Niche ({exemplar})"
            meta["tfidf_scores"] = {}
            meta["degenerate"] = True
            neuron_info[neuron_idx] = meta
            neuron_labels[neuron_idx] = meta["label"]
            continue

        counter = neuron_attrs[neuron_idx]
        total_attr_mentions = sum(counter.values()) or 1

        scored = {}
        for attr_key, tf_raw in counter.items():
            if tf_raw < MIN_ATTR_SUPPORT:
                continue
            cat = attr_key.split(":", 1)[0]
            tf = tf_raw / total_attr_mentions
            idf = attr_idf.get(attr_key, 0)
            bonus = CATEGORY_BONUS.get(cat, 1.0)
            scored[attr_key] = tf * idf * bonus

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)

        labels = []
        used_categories = set()
        for attr_key, score in ranked:
            if score <= 0:
                break
            cat, val = attr_key.split(":", 1)
            if cat in used_categories:
                continue
            labels.append(val)
            used_categories.add(cat)
            if len(labels) >= 2:
                break

        if not labels:
            if meta["genres"]:
                labels.append(next(iter(meta["genres"])))
            else:
                labels.append("Mixed")

        label = " \u2022 ".join(labels)
        meta["label"] = label
        meta["tfidf_scores"] = {k: round(v, 4) for k, v in ranked[:10]}

        neuron_info[neuron_idx] = meta
        neuron_labels[neuron_idx] = label

    # Save
    n_meaningful = sum(
        1 for info in neuron_info.values()
        if info.get("selective") and not info.get("degenerate")
    )
    print(f"\nLabeled {len(neuron_labels)} neurons ({n_meaningful} meaningful, {n_degenerate} niche/degenerate)")

    output_path = DATA_DIR / "neuron_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(neuron_info, f, indent=2, ensure_ascii=False)
    print(f"Full analysis: {output_path}")

    labels_path = DATA_DIR / "neuron_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(neuron_labels, f, indent=2, ensure_ascii=False)
    print(f"Labels: {labels_path}")

    # Print meaningful neurons sorted by activation count (most useful for steering)
    print("\n" + "=" * 80)
    print("TOP MEANINGFUL NEURONS (sorted by activation breadth)")
    print("=" * 80)

    meaningful = [
        (idx, info) for idx, info in neuron_info.items()
        if info.get("selective") and not info.get("degenerate")
    ]
    meaningful.sort(key=lambda x: x[1]["activation_count"], reverse=True)

    for neuron_idx, info in meaningful[:40]:
        n_mov = info["activation_count"]
        sel_pct = info["selectivity"] * 100
        top_tfidf = list(info.get("tfidf_scores", {}).items())[:3]
        exemplars = ", ".join(m["title"][:28] for m in info["top_movies"][:3])
        print(f"\n  N{neuron_idx}: {info['label']}  ({n_mov} movies, {sel_pct:.0f}% sel)")
        if top_tfidf:
            print(f"    TF-IDF: {top_tfidf}")
        print(f"    e.g. {exemplars}")

    return neuron_info, neuron_labels


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-aware", action="store_true",
                        help="Use prediction-aware SAE instead of basic SAE")
    args = parser.parse_args()
    analyze_all_neurons(use_prediction_aware=args.prediction_aware)
