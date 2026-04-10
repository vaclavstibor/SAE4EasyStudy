"""
Dynamic Neuron Labeling for SAE models.

Instead of relying on static neuron_labels.json (which only covers one model),
this module derives human-readable labels from the actual learned activations
of *any* SAE model.

Algorithm (per neuron):
  1. Collect top-K activating movies (the ones where neuron fires strongest).
  2. For those movies, aggregate: (a) genre frequency, (b) MovieLens genome-tag
     relevance scores.
  3. Compute a TF-IDF-like distinctiveness vs. the global average.
  4. Combine the most distinctive tags into a concise label.

Cached per model so the expensive step runs only once.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOP_MOVIES_PER_NEURON = 100  # top-activating movies used for label derivation
MIN_ACTIVATION_COUNT = 20    # neurons with fewer active items get generic label
MAX_LABEL_PARTS = 3          # max concept tokens in one label
CACHE_VERSION = "v2"         # bump to invalidate caches

# Tags to suppress from labels (uninformative or offensive)
BLOCKED_TAGS = {
    'nudity', 'sex', 'prostitution', 'erotic', 'erotica', 'softcore',
    'rape', 'based on a book', 'based on a true story', 'based on novel',
    'original screenplay', 'adult comedy', 'sex comedy',
    'abortion', 'skinhead', 'neo-nazis', 'voodoo', 'nuclear war',
}

DECADE_TAGS = {'1920s', '1930s', '1950s', '1960s', '1970s', '1980s', '1990s'}

# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def get_dynamic_labels(
    model_id: str,
    item_features: torch.Tensor,
    item_ids: list,
    data_dir: str = None,
    force_recompute: bool = False,
) -> Dict[int, dict]:
    """
    Return neuron labels for *any* SAE model, derived from its activations.

    Returns dict:  neuron_id -> {
        'label': str,          # e.g. "Sci-Fi · Space"
        'tags': [str, ...],    # ordered distinctive tags
        'genres': [str, ...],  # ordered distinctive genres
        'activation_count': int,
        'selectivity': float,  # fraction of items where neuron fires
    }

    Results are cached to ``<data_dir>/dynamic_labels_<model_id>.json``.
    """
    if data_dir is None:
        data_dir = str(Path(__file__).parent / "data")

    cache_path = os.path.join(data_dir, f"dynamic_labels_{model_id}_{CACHE_VERSION}.json")

    if not force_recompute and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                raw = json.load(f)
            # Convert string keys back to int
            labels = {int(k): v for k, v in raw.items()}
            print(f"[DynamicLabeling] Loaded cached labels for {model_id}: {len(labels)} neurons")
            return labels
        except Exception as e:
            print(f"[DynamicLabeling] Cache load failed: {e}")

    print(f"[DynamicLabeling] Computing labels for model={model_id} "
          f"(features shape={item_features.shape})...")

    # --- Load MovieLens metadata -------------------------------------------
    ml_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'datasets', 'ml-latest')
    movies_genres = _load_movie_genres(ml_dir)         # movieId -> [genre, ...]
    genome_tags, genome_scores = _load_genome(ml_dir)  # tagId->name, movieId->{tagId: relevance}

    # --- Global tag distribution (baseline) --------------------------------
    global_genre_freq = _global_genre_freq(movies_genres)
    global_tag_avg = _global_tag_avg(genome_scores, genome_tags)

    # --- Per-neuron analysis -----------------------------------------------
    features_np = item_features.cpu().numpy() if isinstance(item_features, torch.Tensor) else item_features
    n_items, n_neurons = features_np.shape
    item_ids_list = [int(x) for x in item_ids]

    labels = {}
    for nid in range(n_neurons):
        col = features_np[:, nid]
        activation_count = int(np.sum(col > 0))

        if activation_count < MIN_ACTIVATION_COUNT:
            continue  # skip dead/near-dead neurons

        selectivity = 1.0 - activation_count / n_items

        # Top-activating movies
        top_k = min(TOP_MOVIES_PER_NEURON, activation_count)
        top_indices = np.argpartition(col, -top_k)[-top_k:]
        top_movie_ids = [item_ids_list[i] for i in top_indices]
        top_weights = col[top_indices]
        # Normalise weights to sum to 1
        w_sum = top_weights.sum()
        if w_sum > 0:
            top_weights = top_weights / w_sum
        else:
            top_weights = np.ones(len(top_weights)) / len(top_weights)

        # --- Aggregate genres (weighted) ---
        genre_scores = {}
        for mid, w in zip(top_movie_ids, top_weights):
            for g in movies_genres.get(mid, []):
                genre_scores[g] = genre_scores.get(g, 0) + w

        # TF-IDF-like: ratio of local frequency to global frequency
        genre_tfidf = {}
        for g, score in genre_scores.items():
            gf = global_genre_freq.get(g, 0.01)
            genre_tfidf[g] = score / gf

        # --- Aggregate genome tags (weighted) ---
        tag_scores = {}
        for mid, w in zip(top_movie_ids, top_weights):
            mid_tags = genome_scores.get(mid, {})
            for tid, relevance in mid_tags.items():
                tag_scores[tid] = tag_scores.get(tid, 0) + w * relevance

        # TF-IDF-like: ratio to global average
        tag_tfidf = {}
        for tid, score in tag_scores.items():
            tag_name = genome_tags.get(tid, '')
            if tag_name.lower() in BLOCKED_TAGS:
                continue
            if tag_name.lower() in DECADE_TAGS:
                continue
            ga = global_tag_avg.get(tid, 0.01)
            tag_tfidf[tid] = score / max(ga, 0.001)

        # --- Build label from top distinctive tags + genres -----------------
        # Sort genres by distinctiveness
        sorted_genres = sorted(genre_tfidf.items(), key=lambda x: -x[1])
        # Sort tags by distinctiveness
        sorted_tags = sorted(tag_tfidf.items(), key=lambda x: -x[1])

        # Pick the most distinctive tag names — avoid near-duplicates
        top_tag_names = []
        used_words = set()
        genre_names_lower = {g.lower() for g in genre_scores}
        for tid, score in sorted_tags:
            name = genome_tags.get(tid, '')
            if not name or name.lower() in genre_names_lower:
                continue  # skip if it's just a genre name
            if len(name) < 2:
                continue
            # Avoid near-duplicate by checking word overlap
            words = set(name.lower().split())
            if words & used_words:
                continue
            top_tag_names.append(name)
            used_words |= words
            if len(top_tag_names) >= 5:
                break

        # Only keep the most distinctive genre (skip IMAX, it's not a real genre)
        FILLER_GENRES = {'imax', '(no genres listed)'}
        top_genre_names = [g for g, _ in sorted_genres[:3]
                           if g.lower() not in FILLER_GENRES]

        # Combine: prefer concept tags; only add genre if we need more parts
        label_parts = []
        for t in top_tag_names[:2]:
            label_parts.append(t.title() if len(t) > 3 else t)

        # Add genre only if we have fewer than 2 concept tags
        if len(label_parts) < 2:
            for g in top_genre_names:
                if g.lower() not in {p.lower() for p in label_parts}:
                    label_parts.append(g)
                if len(label_parts) >= 2:
                    break

        if not label_parts:
            label_parts = [f"Feature {nid}"]

        label = ' · '.join(label_parts[:MAX_LABEL_PARTS])

        labels[nid] = {
            'label': label,
            'tags': top_tag_names[:5],
            'genres': top_genre_names[:3],
            'activation_count': activation_count,
            'selectivity': round(selectivity, 4),
        }

    # --- Cache results ----------------------------------------------------
    try:
        with open(cache_path, 'w') as f:
            json.dump({str(k): v for k, v in labels.items()}, f, indent=1)
        print(f"[DynamicLabeling] Cached {len(labels)} labels to {cache_path}")
    except Exception as e:
        print(f"[DynamicLabeling] Cache write failed: {e}")

    return labels


# ---------------------------------------------------------------------------
# Select best neurons for display
# ---------------------------------------------------------------------------

def select_features_for_display(
    dynamic_labels: Dict[int, dict],
    top_k: int = 21,
) -> List[dict]:
    """
    From the full set of dynamically labeled neurons, pick the best ``top_k``
    for the steering UI.

    Scoring: selectivity × log(activation_count) × concept_bonus
    Diversity: no duplicate top-tag, limit pure-genre neurons.
    """

    GENRE_NAMES = {
        'action', 'adventure', 'animation', 'children', 'comedy', 'crime',
        'documentary', 'drama', 'fantasy', 'film-noir', 'horror', 'imax',
        'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war',
        'western', 'family',
    }

    candidates = []
    for nid, info in dynamic_labels.items():
        act_count = info['activation_count']
        sel = info['selectivity']
        tags = info.get('tags', [])
        genres = info.get('genres', [])
        label = info['label']

        if act_count < 50 or act_count > 40000:
            continue
        if sel < 0.3:
            continue

        has_concept = any(t.lower() not in GENRE_NAMES and t.lower() not in DECADE_TAGS
                         for t in tags[:2])
        concept_bonus = 2.0 if has_concept else 1.0
        score = sel * np.log(act_count + 1) * concept_bonus

        candidates.append({
            'neuron_id': nid,
            'label': label,
            'tags': tags,
            'genres': genres,
            'score': score,
            'has_concept': has_concept,
            'activation_count': act_count,
            'selectivity': sel,
        })

    candidates.sort(key=lambda x: -x['score'])

    selected = []
    used_words = {}    # individual word -> count
    MAX_WORD_USES = 2  # allow each meaningful word at most twice
    STOP_WORDS = {
        'the', 'and', 'of', 'in', 'a', 'an', 'to', 'for', 'with', 'on',
        'at', 'by', 'from', 'its', 'that', 'this', 'but', 'or', 'as',
    }
    used_genres = set()
    pure_genre_count = 0
    MAX_PURE_GENRE = 7

    for c in candidates:
        if len(selected) >= top_k:
            break

        # Extract individual words from label for diversity checking.
        # Split on both '·' (TF-IDF labels) and whitespace (LLM labels).
        import re as _re
        label_words = {
            w for w in _re.split(r'[\s·\-–—/,&]+', c['label'].lower())
            if len(w) > 2 and w not in STOP_WORDS
        }

        # Diversity: skip if ANY meaningful word is already overused
        if any(used_words.get(w, 0) >= MAX_WORD_USES for w in label_words):
            continue

        # Limit pure-genre neurons
        if not c['has_concept']:
            if pure_genre_count >= MAX_PURE_GENRE:
                continue
            g_set = {g.lower() for g in c['genres']}
            if g_set and g_set <= used_genres:
                continue

        selected.append(c)
        for w in label_words:
            used_words[w] = used_words.get(w, 0) + 1
        for g in c['genres']:
            used_genres.add(g.lower())
        if not c['has_concept']:
            pure_genre_count += 1

    features = []
    for s in selected:
        nid = s['neuron_id']
        features.append({
            'id': nid,
            'label': s['label'],
            'category': 'latent',
            'description': f"Latent concept (N{nid}): {s['label']}",
            'top_tags': [],
            'activation': 0.5,
            'movie_count': s['activation_count'],
        })

    features.sort(key=lambda f: -f['movie_count'])

    print(f"[DynamicLabeling] Selected {len(features)} features for display:")
    for f in features:
        print(f"  N{f['id']:5d}  cnt={f['movie_count']:5d}  {f['label']}")

    return features


# ---------------------------------------------------------------------------
# Label specific neurons (even those below thresholds)
# ---------------------------------------------------------------------------

def label_neurons_by_ids(
    neuron_ids: List[int],
    model_id: str,
    item_features,  # torch.Tensor or numpy array
    item_ids: list,
    data_dir: str = None,
) -> Dict[int, dict]:
    """
    Compute or retrieve labels for *specific* neuron IDs.

    Unlike get_dynamic_labels() which skips neurons below MIN_ACTIVATION_COUNT,
    this function will produce a label for ANY neuron that has at least 1
    activating item, using the same TF-IDF tag/genre algorithm.

    Already-cached labels are reused; newly computed ones are appended to the
    cache file.
    """
    if data_dir is None:
        data_dir = str(Path(__file__).parent / "data")

    cache_path = os.path.join(data_dir, f"dynamic_labels_{model_id}_{CACHE_VERSION}.json")

    # Load existing cache
    existing: Dict[int, dict] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                raw = json.load(f)
            existing = {int(k): v for k, v in raw.items()}
        except Exception:
            pass

    # Find which neurons are missing
    missing = [nid for nid in neuron_ids if nid not in existing]
    if not missing:
        return {nid: existing[nid] for nid in neuron_ids if nid in existing}

    print(f"[DynamicLabeling] Computing labels for {len(missing)} missing neurons: {missing}")

    # --- Load metadata (same as get_dynamic_labels) ---
    ml_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'datasets', 'ml-latest')
    movies_genres = _load_movie_genres(ml_dir)
    genome_tags, genome_scores = _load_genome(ml_dir)
    global_genre_freq = _global_genre_freq(movies_genres)
    global_tag_avg = _global_tag_avg(genome_scores, genome_tags)

    features_np = item_features.cpu().numpy() if isinstance(item_features, torch.Tensor) else item_features
    n_items, n_neurons = features_np.shape
    item_ids_list = [int(x) for x in item_ids]

    new_labels = {}
    for nid in missing:
        if nid < 0 or nid >= n_neurons:
            continue
        col = features_np[:, nid]
        activation_count = int(np.sum(col > 0))

        if activation_count == 0:
            new_labels[nid] = {
                'label': f'Feature {nid}',
                'tags': [],
                'genres': [],
                'activation_count': 0,
                'selectivity': 1.0,
            }
            continue

        selectivity = 1.0 - activation_count / n_items
        top_k = min(TOP_MOVIES_PER_NEURON, activation_count)
        top_indices = np.argpartition(col, -top_k)[-top_k:]
        top_movie_ids = [item_ids_list[i] for i in top_indices]
        top_weights = col[top_indices]
        w_sum = top_weights.sum()
        if w_sum > 0:
            top_weights = top_weights / w_sum
        else:
            top_weights = np.ones(len(top_weights)) / len(top_weights)

        # Aggregate genres
        genre_scores_d: Dict[str, float] = {}
        for mid, w in zip(top_movie_ids, top_weights):
            for g in movies_genres.get(mid, []):
                genre_scores_d[g] = genre_scores_d.get(g, 0) + w
        genre_tfidf = {}
        for g, score in genre_scores_d.items():
            gf = global_genre_freq.get(g, 0.01)
            genre_tfidf[g] = score / gf

        # Aggregate genome tags
        tag_scores_d: Dict[int, float] = {}
        for mid, w in zip(top_movie_ids, top_weights):
            mid_tags = genome_scores.get(mid, {})
            for tid, relevance in mid_tags.items():
                tag_scores_d[tid] = tag_scores_d.get(tid, 0) + w * relevance
        tag_tfidf = {}
        for tid, score in tag_scores_d.items():
            tag_name = genome_tags.get(tid, '')
            if tag_name.lower() in BLOCKED_TAGS or tag_name.lower() in DECADE_TAGS:
                continue
            ga = global_tag_avg.get(tid, 0.01)
            tag_tfidf[tid] = score / max(ga, 0.001)

        sorted_genres = sorted(genre_tfidf.items(), key=lambda x: -x[1])
        sorted_tags = sorted(tag_tfidf.items(), key=lambda x: -x[1])

        top_tag_names = []
        used_words: set = set()
        genre_names_lower = {g.lower() for g in genre_scores_d}
        for tid, score in sorted_tags:
            name = genome_tags.get(tid, '')
            if not name or name.lower() in genre_names_lower:
                continue
            if len(name) < 2:
                continue
            words = set(name.lower().split())
            if words & used_words:
                continue
            top_tag_names.append(name)
            used_words |= words
            if len(top_tag_names) >= 5:
                break

        FILLER_GENRES = {'imax', '(no genres listed)'}
        top_genre_names = [g for g, _ in sorted_genres[:3]
                           if g.lower() not in FILLER_GENRES]

        label_parts = []
        for t in top_tag_names[:2]:
            label_parts.append(t.title() if len(t) > 3 else t)
        if len(label_parts) < 2:
            for g in top_genre_names:
                if g.lower() not in {p.lower() for p in label_parts}:
                    label_parts.append(g)
                if len(label_parts) >= 2:
                    break
        if not label_parts:
            label_parts = [f"Feature {nid}"]

        label = ' · '.join(label_parts[:MAX_LABEL_PARTS])
        new_labels[nid] = {
            'label': label,
            'tags': top_tag_names[:5],
            'genres': top_genre_names[:3],
            'activation_count': activation_count,
            'selectivity': round(selectivity, 4),
        }
        print(f"  N{nid}: act={activation_count}, label=\"{label}\"")

    # Append new labels to cache
    if new_labels:
        merged = {**existing, **new_labels}
        try:
            with open(cache_path, 'w') as f:
                json.dump({str(k): v for k, v in merged.items()}, f, indent=1)
            print(f"[DynamicLabeling] Updated cache with {len(new_labels)} new labels "
                  f"(total: {len(merged)})")
        except Exception as e:
            print(f"[DynamicLabeling] Cache update failed: {e}")

    # Return labels for all requested neurons
    all_labels = {**existing, **new_labels}
    return {nid: all_labels[nid] for nid in neuron_ids if nid in all_labels}


# ---------------------------------------------------------------------------
# Internal helpers for loading MovieLens metadata
# ---------------------------------------------------------------------------

def _load_movie_genres(ml_dir: str) -> Dict[int, List[str]]:
    """Parse movies.csv → {movieId: [genre, ...]}"""
    import csv
    movies_path = os.path.join(ml_dir, 'movies.csv')
    result = {}
    if not os.path.exists(movies_path):
        print(f"[DynamicLabeling] movies.csv not found at {movies_path}")
        return result
    with open(movies_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = int(row['movieId'])
            genres = row['genres'].split('|')
            result[mid] = genres
    return result


def _load_genome(ml_dir: str) -> Tuple[Dict[int, str], Dict[int, Dict[int, float]]]:
    """
    Load genome-tags.csv and genome-scores.csv.

    Returns:
        genome_tags:   {tagId: tag_name}
        genome_scores: {movieId: {tagId: relevance}}
    """
    import csv
    tags_path = os.path.join(ml_dir, 'genome-tags.csv')
    scores_path = os.path.join(ml_dir, 'genome-scores.csv')

    genome_tags = {}
    if os.path.exists(tags_path):
        with open(tags_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                genome_tags[int(row['tagId'])] = row['tag']
    else:
        print(f"[DynamicLabeling] genome-tags.csv not found at {tags_path}")

    genome_scores = {}
    if os.path.exists(scores_path):
        with open(scores_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                mid = int(row['movieId'])
                tid = int(row['tagId'])
                rel = float(row['relevance'])
                if rel < 0.1:
                    continue  # skip very low relevance to save memory
                if mid not in genome_scores:
                    genome_scores[mid] = {}
                genome_scores[mid][tid] = rel
    else:
        print(f"[DynamicLabeling] genome-scores.csv not found at {scores_path}")

    print(f"[DynamicLabeling] Loaded {len(genome_tags)} genome tags, "
          f"{len(genome_scores)} movies with genome scores")
    return genome_tags, genome_scores


def _global_genre_freq(movies_genres: Dict[int, List[str]]) -> Dict[str, float]:
    """Compute global genre frequency (fraction of movies having each genre)."""
    total = len(movies_genres)
    if total == 0:
        return {}
    counts = {}
    for genres in movies_genres.values():
        for g in genres:
            counts[g] = counts.get(g, 0) + 1
    return {g: c / total for g, c in counts.items()}


def _global_tag_avg(
    genome_scores: Dict[int, Dict[int, float]],
    genome_tags: Dict[int, str],
) -> Dict[int, float]:
    """Compute average genome-tag relevance across all movies."""
    tag_sums = {}
    tag_counts = {}
    for mid, tags in genome_scores.items():
        for tid, rel in tags.items():
            tag_sums[tid] = tag_sums.get(tid, 0) + rel
            tag_counts[tid] = tag_counts.get(tid, 0) + 1
    total_movies = len(genome_scores)
    if total_movies == 0:
        return {}
    # Average over ALL movies (not just those with the tag)
    return {tid: s / total_movies for tid, s in tag_sums.items()}
