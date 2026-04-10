"""
LLM-based Neuron Labeling for SAE models.

Follows the RecSAE (Wang et al., 2024) pipeline:
  1. For each SAE neuron, find the top-N activating items.
  2. Collect item metadata (title, genres, year, genome tags).
  3. Prompt a local LLM (Llama-3-8B-Instruct via llama-cpp-python or
     an OpenAI-compatible API) to generate a concept description.
  4. Structured JSON output: short UI label + user-facing description.

LLM is the ONLY labeling path. Pre-generate labels offline before deploying
using generate_llm_labels.py. If the LLM is unreachable at runtime, cached
labels are used; if no cache exists, a RuntimeError is raised.

Usage:
    from .llm_labeling import get_llm_labels
    labels = get_llm_labels(model_id, item_features, item_ids)
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOP_N_ITEMS = 30          # items to show the LLM per neuron
MIN_ACTIVATION_COUNT = 20 # skip near-dead neurons
CACHE_VERSION = "llm"  # single version — always overwrite
MAX_RETRIES = 3

# LLM connection settings — override via environment variables
# Default: Ollama running locally (no API key needed)
LLM_BACKEND = os.environ.get("SAE_LLM_BACKEND", "openai")  # "openai" | "llamacpp"
LLM_MODEL = os.environ.get("SAE_LLM_MODEL", "llama3:8b-instruct-q4_0")
LLM_API_BASE = os.environ.get("SAE_LLM_API_BASE", "http://localhost:11434/v1")
LLM_API_KEY = os.environ.get("SAE_LLM_API_KEY", "not-needed")
# For llama-cpp-python local: path to GGUF file
LLM_GGUF_PATH = os.environ.get("SAE_LLM_GGUF_PATH", "")

# ---------------------------------------------------------------------------
# Comprehensive LLM Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert film analyst working on an interactive movie recommender system.
Your task is to examine a cluster of movies that share a latent concept learned
by a neural recommendation model. From the titles, genres, years, and content
tags you must produce TWO things:

1. **label** — a short, user-friendly name (3-4 words) for a UI slider that
   controls how much a user wants this kind of movie. The label must be
   immediately understandable to an average movie-goer who has never studied
   film. No jargon, no abstract words. Use Title Case. Think of it as a
   shelf label in a video store that anyone can read and instantly know
   what kind of movies are there.
   BAD labels: "Emotional Journeys", "Visceral Thrill Rides", "Whimsical Storytelling"
   GOOD labels: "Sad Family Movies", "Fast Action Blockbusters", "Weird Indie Comedies"

2. **description** — one or two sentences describing the mood, theme, or style
   that connects these movies. Focus on what the viewer *experiences* or
   *feels* — atmosphere, pacing, emotional tone, visual style, recurring themes.
   Do NOT mention specific movie titles. Write in plain language.

Reply with ONLY valid JSON (no markdown fences, no extra text):
{"label": "...", "description": "..."}
"""

USER_PROMPT_TEMPLATE = """\
Below are the {n_movies} movies that most strongly activate a particular latent
dimension in the recommendation model.

Activation strength indicates how representative each movie is for this dimension.

Movies (sorted by activation strength, strongest first):
{movie_list}

Most common genres across these movies: {genre_summary}
Most common content tags: {tag_summary}

Task:
Identify the SINGLE most specific unifying concept that explains why these
movies cluster together.

Prioritize:
- theme
- tone / mood
- visual style
- narrative archetype
- director / franchise
- era or cultural movement

Avoid generic genre labels unless you add a meaningful qualifier.

Description rules:
- Write a **concise atmospheric description** of the *type of movie experience*.
- Describe the **tone, themes, and storytelling style**.
- DO NOT refer to the movies directly.
- DO NOT start with phrases like:
  "These movies", "These films", "A collection of", "This set of".
- The text should read like the explanation of a **recommendation slider**.

Good example description:
"Slow-burn psychological tension built around obsession, moral ambiguity,
and characters pushed to emotional extremes."

Bad example:
"These movies explore complex relationships and emotional journeys."

Reply with ONLY valid JSON:
{{"label": "...", "description": "..."}}
"""


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Thin wrapper around local LLM inference (OpenAI-compatible API or llama-cpp)."""

    def __init__(self):
        self._client = None
        self._llama = None
        self._backend = LLM_BACKEND

    def _init_openai(self):
        """Initialize OpenAI-compatible client (works with vllm, llama.cpp server, etc.)."""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=LLM_API_BASE,
                api_key=LLM_API_KEY,
            )
            return True
        except ImportError:
            print("[LLM] openai package not installed. pip install openai")
            return False

    def _init_llamacpp(self):
        """Initialize llama-cpp-python for local inference."""
        try:
            from llama_cpp import Llama
            if not LLM_GGUF_PATH or not os.path.exists(LLM_GGUF_PATH):
                print(f"[LLM] GGUF model not found at {LLM_GGUF_PATH}")
                return False
            self._llama = Llama(
                model_path=LLM_GGUF_PATH,
                n_ctx=2048,
                n_threads=4,
                verbose=False,
            )
            return True
        except ImportError:
            print("[LLM] llama-cpp-python not installed. pip install llama-cpp-python")
            return False

    def generate(self, system: str, user: str, max_tokens: int = 120) -> Optional[str]:
        """Generate a response from the LLM."""
        if self._backend == "openai":
            if self._client is None:
                if not self._init_openai():
                    return None
            try:
                resp = self._client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.15,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[LLM] OpenAI API error: {e}")
                return None

        if self._backend == "llamacpp":
            if self._llama is None:
                if not self._init_llamacpp():
                    return None
            try:
                resp = self._llama.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.15,
                )
                return resp["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"[LLM] llama-cpp error: {e}")
                return None

        return None

    def is_available(self) -> bool:
        """Quick check if the LLM backend is reachable."""
        result = self.generate("Say OK.", "Test", max_tokens=5)
        return result is not None


# Singleton
_llm_client = None

def _get_llm() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


# ---------------------------------------------------------------------------
# Metadata loading (reuse from dynamic_labeling where possible)
# ---------------------------------------------------------------------------

def _load_movie_metadata(ml_dir: str) -> Dict[int, dict]:
    """Load movie metadata: title, genres, year.

    Returns: {movieId: {'title': str, 'genres': [str], 'year': int|None}}
    """
    import csv
    import re

    movies_path = os.path.join(ml_dir, 'movies.csv')
    result = {}
    if not os.path.exists(movies_path):
        return result

    year_re = re.compile(r'\((\d{4})\)\s*$')
    with open(movies_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            mid = int(row['movieId'])
            title = row['title'].strip()
            genres = row['genres'].split('|')
            year_match = year_re.search(title)
            year = int(year_match.group(1)) if year_match else None
            result[mid] = {
                'title': title,
                'genres': genres,
                'year': year,
            }
    return result


def _load_genome_tags_for_movie(ml_dir: str) -> Tuple[Dict[int, str], Dict[int, List[str]]]:
    """Load genome tags and return top-5 tags per movie.

    Returns:
        genome_tags: {tagId: tag_name}
        movie_top_tags: {movieId: [tag_name, ...]}  (top-5 by relevance)
    """
    import csv

    tags_path = os.path.join(ml_dir, 'genome-tags.csv')
    scores_path = os.path.join(ml_dir, 'genome-scores.csv')

    genome_tags = {}
    if os.path.exists(tags_path):
        with open(tags_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                genome_tags[int(row['tagId'])] = row['tag']

    # Collect top-5 tags per movie
    movie_scores: Dict[int, List[Tuple[float, int]]] = {}
    if os.path.exists(scores_path):
        with open(scores_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                mid = int(row['movieId'])
                tid = int(row['tagId'])
                rel = float(row['relevance'])
                if rel < 0.5:
                    continue
                if mid not in movie_scores:
                    movie_scores[mid] = []
                movie_scores[mid].append((rel, tid))

    movie_top_tags = {}
    for mid, scores in movie_scores.items():
        scores.sort(reverse=True)
        movie_top_tags[mid] = [
            genome_tags.get(tid, '') for _, tid in scores[:5]
            if genome_tags.get(tid, '')
        ]

    return genome_tags, movie_top_tags


# ---------------------------------------------------------------------------
# Core labeling pipeline
# ---------------------------------------------------------------------------

def _format_movie_list(
    movie_ids: List[int],
    weights: np.ndarray,
    metadata: Dict[int, dict],
    top_tags: Dict[int, List[str]],
    max_items: int = TOP_N_ITEMS,
) -> str:
    """Format a list of movies for the LLM prompt with rich metadata."""
    lines = []
    for idx, (mid, w) in enumerate(zip(movie_ids[:max_items], weights[:max_items]), 1):
        info = metadata.get(mid)
        if not info:
            continue
        title = info['title']
        genres = ', '.join(g for g in info['genres'] if g != '(no genres listed)')
        year = info.get('year')
        year_str = f" ({year})" if year else ""
        tags = top_tags.get(mid, [])
        tag_str = f"  Tags: {', '.join(tags[:4])}" if tags else ""
        lines.append(f"  {idx}. {title}{year_str}  [{genres}]{tag_str}  (activation: {w:.3f})")
    return '\n'.join(lines)


def _parse_llm_json(raw: str) -> Optional[dict]:
    """Robustly parse the LLM's JSON response.

    Handles markdown fences, trailing text, and minor formatting issues.
    """
    if not raw:
        return None
    # Strip markdown code fences if present
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned)
    # Try direct parse
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and 'label' in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # Try to find JSON object in the text
    match = re.search(r'\{[^{}]*"label"\s*:\s*"[^"]+?"[^{}]*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Last resort: extract label from raw text
    label_match = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
    desc_match = re.search(r'"description"\s*:\s*"([^"]+)"', raw)
    if label_match:
        return {
            'label': label_match.group(1),
            'description': desc_match.group(1) if desc_match else '',
        }
    return None


def label_neuron_with_llm(
    neuron_id: int,
    top_movie_ids: List[int],
    top_weights: np.ndarray,
    metadata: Dict[int, dict],
    top_tags: Dict[int, List[str]],
    genre_summary: str = "",
    tag_summary: str = "",
) -> Optional[dict]:
    """Use LLM to generate a concept label + description for one neuron.

    Returns: {'label': str, 'description': str} or None on failure.
    """
    llm = _get_llm()
    movie_list = _format_movie_list(top_movie_ids, top_weights, metadata, top_tags)
    if not movie_list:
        return None

    prompt = USER_PROMPT_TEMPLATE.format(
        n_movies=len(top_movie_ids[:TOP_N_ITEMS]),
        movie_list=movie_list,
        genre_summary=genre_summary or "N/A",
        tag_summary=tag_summary or "N/A",
    )

    for attempt in range(MAX_RETRIES + 1):
        raw = llm.generate(SYSTEM_PROMPT, prompt, max_tokens=200)
        if raw:
            parsed = _parse_llm_json(raw)
            if parsed:
                label = parsed['label'].strip().strip('"\'')
                description = parsed.get('description', '').strip().strip('"\'')
                if 2 <= len(label) <= 60:
                    return {'label': label, 'description': description}
            # If JSON parse fails, try to use raw text as label (old-style)
            clean = raw.strip().strip('"\'').split('\n')[0].strip()
            if 2 <= len(clean) <= 60:
                return {
                    'label': clean,
                    'description': f"More movies like these: {clean}",
                }
        if attempt < MAX_RETRIES:
            time.sleep(0.5 * (attempt + 1))

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_llm_labels(
    model_id: str,
    item_features,  # torch.Tensor or numpy
    item_ids: list,
    data_dir: str = None,
    force_recompute: bool = False,
    offline_mode: bool = False,
) -> Dict[int, dict]:
    """
    Generate LLM-based concept labels for all active SAE neurons.

    Args:
        offline_mode: If True, skip cache early-return and run the full
            labeling loop with resume support. Used by generate_llm_labels.py.
            If False (default/runtime), return cache immediately.

    Returns dict: neuron_id -> {
        'label': str,            # Short UI label (2-6 words)
        'description': str,      # User-facing sentence for tooltip
        'label_source': 'llm',
        'tags': [str, ...],
        'genres': [str, ...],
        'activation_count': int,
        'selectivity': float,
        'top_movies': [int, ...],
    }
    """
    import torch

    if data_dir is None:
        data_dir = str(Path(__file__).parent / "data")

    cache_path = os.path.join(data_dir, f"llm_labels_{model_id}_{CACHE_VERSION}.json")

    # Cache first — at runtime, always return immediately.
    # In offline_mode (generate_llm_labels.py), skip this and enter labeling loop.
    if not force_recompute and not offline_mode and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                raw = json.load(f)
            cached_labels = {int(k): v for k, v in raw.items()}
            print(f"[LLM-Labeling] Loaded {len(cached_labels)} cached LLM labels for {model_id}")
            return cached_labels
        except Exception as e:
            print(f"[LLM-Labeling] Cache load failed: {e}")

    # LLM is mandatory — check availability
    llm = _get_llm()
    if not llm.is_available():
        raise RuntimeError(
            "[LLM-Labeling] LLM is unreachable and no label cache exists. "
            "Please run `python generate_llm_labels.py --model {model_id}` "
            "to pre-generate labels before starting the server."
        )

    print(f"[LLM-Labeling] LLM available ({LLM_BACKEND}), computing labels for {model_id}...")

    # Load metadata
    ml_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'datasets', 'ml-latest')
    metadata = _load_movie_metadata(ml_dir)
    _, movie_top_tags = _load_genome_tags_for_movie(ml_dir)

    # Prepare features
    features_np = item_features.cpu().numpy() if isinstance(item_features, torch.Tensor) else item_features
    n_items, n_neurons = features_np.shape
    item_ids_list = [int(x) for x in item_ids]

    # Load existing partial cache (for resume after interruption)
    labels = {}
    if not force_recompute and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                labels = {int(k): v for k, v in json.load(f).items()}
            print(f"[LLM-Labeling] Resuming: {len(labels)} neurons already cached")
        except Exception:
            pass

    labeled_count = 0
    skipped_count = 0
    failed_count = 0
    resumed_count = len(labels)
    start_time = time.time()

    # Pre-count active neurons for progress reporting
    total_active = 0
    for _nid in range(n_neurons):
        if _nid not in labels:
            _act = int(np.sum(features_np[:, _nid] > 0))
            if _act >= MIN_ACTIVATION_COUNT:
                total_active += 1
    print(f"[LLM-Labeling] {total_active} active neurons to label "
          f"(+ {resumed_count} already cached, {n_neurons} total)")

    for nid in range(n_neurons):
        # Skip already-labeled neurons (resume support)
        if nid in labels:
            continue

        col = features_np[:, nid]
        activation_count = int(np.sum(col > 0))

        if activation_count < MIN_ACTIVATION_COUNT:
            skipped_count += 1
            continue

        selectivity = 1.0 - activation_count / n_items

        # Top-activating movies
        top_k = min(TOP_N_ITEMS, activation_count)
        top_indices = np.argpartition(col, -top_k)[-top_k:]
        sorted_order = np.argsort(-col[top_indices])
        top_indices = top_indices[sorted_order]
        top_movie_ids = [item_ids_list[i] for i in top_indices]
        top_weights = col[top_indices]

        # Aggregate genres from top movies
        genre_counts: Dict[str, int] = {}
        for mid in top_movie_ids[:TOP_N_ITEMS]:
            info = metadata.get(mid)
            if info:
                for g in info['genres']:
                    if g != '(no genres listed)':
                        genre_counts[g] = genre_counts.get(g, 0) + 1
        genres = sorted(genre_counts.keys(), key=lambda g: -genre_counts[g])[:5]

        # Aggregate tags from top movies
        tag_counts: Dict[str, int] = {}
        for mid in top_movie_ids[:TOP_N_ITEMS]:
            for t in movie_top_tags.get(mid, []):
                tag_counts[t] = tag_counts.get(t, 0) + 1
        tags = sorted(tag_counts.keys(), key=lambda t: -tag_counts[t])[:7]

        # Build summary strings for prompt
        genre_summary = ', '.join(f"{g} ({genre_counts[g]})" for g in genres[:5])
        tag_summary = ', '.join(f"{t} ({tag_counts[t]})" for t in tags[:5])

        # Call LLM
        llm_result = label_neuron_with_llm(
            neuron_id=nid,
            top_movie_ids=top_movie_ids,
            top_weights=top_weights,
            metadata=metadata,
            top_tags=movie_top_tags,
            genre_summary=genre_summary,
            tag_summary=tag_summary,
        )

        if llm_result:
            label = llm_result['label']
            description = llm_result.get('description', f"More {label.lower()} movies")
        else:
            # LLM failed for this neuron — construct a reasonable label from metadata
            failed_count += 1
            label_parts = []
            if tags:
                label_parts.append(tags[0].title())
            if genres:
                label_parts.append(genres[0])
            label = ' · '.join(label_parts) if label_parts else f"Feature {nid}"
            description = f"More movies in the {label.lower()} space"

        labels[nid] = {
            'label': label,
            'description': description,
            'label_source': 'llm' if llm_result else 'metadata',
            'tags': tags[:5],
            'genres': genres[:3],
            'activation_count': activation_count,
            'selectivity': round(selectivity, 4),
            'top_movies': top_movie_ids[:5],
        }
        labeled_count += 1

        # Per-neuron log (compact, one line)
        src_tag = "✓" if llm_result else "✗"
        print(f"  [{labeled_count}/{total_active}] neuron {nid}: {src_tag} \"{label}\" — {description[:60]}")

        # Detailed progress + incremental save every 50 neurons
        if labeled_count % 50 == 0:
            elapsed = time.time() - start_time
            total_done = labeled_count + resumed_count
            rate = elapsed / labeled_count  # seconds per neuron
            remaining = (total_active - labeled_count) * rate
            pct = labeled_count / total_active * 100 if total_active else 100
            # Format ETA as hours:minutes
            eta_h = int(remaining // 3600)
            eta_m = int((remaining % 3600) // 60)
            eta_str = f"{eta_h}h{eta_m:02d}m" if eta_h > 0 else f"{eta_m}m"
            print(f"[LLM-Labeling] ▸ {labeled_count}/{total_active} ({pct:.1f}%) "
                  f"| {rate:.2f}s/neuron | ETA: {eta_str} | "
                  f"elapsed: {elapsed/60:.1f}min | fails: {failed_count}")
            # Incremental cache save (resume-safe)
            try:
                os.makedirs(data_dir, exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump({str(k): v for k, v in labels.items()}, f, indent=1)
            except Exception:
                pass

    elapsed = time.time() - start_time
    llm_count = sum(1 for v in labels.values() if v['label_source'] == 'llm')
    print(f"[LLM-Labeling] Done: {labeled_count} new + {resumed_count} resumed = {len(labels)} total "
          f"in {elapsed:.1f}s ({llm_count} LLM, {failed_count} metadata-fallback, {skipped_count} skipped)")

    # Cache
    try:
        os.makedirs(data_dir, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({str(k): v for k, v in labels.items()}, f, indent=1)
        print(f"[LLM-Labeling] Cached to {cache_path}")
    except Exception as e:
        print(f"[LLM-Labeling] Cache write failed: {e}")

    return labels


def label_neurons_by_ids_llm(
    neuron_ids: List[int],
    model_id: str,
    item_features,
    item_ids: list,
    data_dir: str = None,
) -> Dict[int, dict]:
    """Label specific neurons using LLM (with cache).

    Checks cache first; labels missing neurons via LLM. No TF-IDF fallback.
    """
    import torch

    if data_dir is None:
        data_dir = str(Path(__file__).parent / "data")

    cache_path = os.path.join(data_dir, f"llm_labels_{model_id}_{CACHE_VERSION}.json")

    # Load existing cache
    existing: Dict[int, dict] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                existing = {int(k): v for k, v in json.load(f).items()}
        except Exception:
            pass

    missing = [nid for nid in neuron_ids if nid not in existing]
    if not missing:
        return {nid: existing[nid] for nid in neuron_ids if nid in existing}

    # LLM is mandatory
    llm = _get_llm()
    if not llm.is_available():
        print("[LLM-Labeling] LLM unreachable — returning cached + placeholder labels")
        result = {}
        for nid in neuron_ids:
            if nid in existing:
                result[nid] = existing[nid]
            else:
                result[nid] = {
                    'label': f'Feature {nid}',
                    'description': f'Latent concept #{nid}',
                    'label_source': 'placeholder',
                    'tags': [], 'genres': [],
                    'activation_count': 0, 'selectivity': 1.0,
                    'top_movies': [],
                }
        return result

    # Load metadata
    ml_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'datasets', 'ml-latest')
    metadata = _load_movie_metadata(ml_dir)
    _, movie_top_tags = _load_genome_tags_for_movie(ml_dir)

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
                'description': f'Latent concept #{nid} (inactive)',
                'label_source': 'placeholder',
                'tags': [], 'genres': [],
                'activation_count': 0, 'selectivity': 1.0,
                'top_movies': [],
            }
            continue

        selectivity = 1.0 - activation_count / n_items
        top_k = min(TOP_N_ITEMS, activation_count)
        top_indices = np.argpartition(col, -top_k)[-top_k:]
        sorted_order = np.argsort(-col[top_indices])
        top_indices = top_indices[sorted_order]
        top_movie_ids = [item_ids_list[i] for i in top_indices]
        top_weights = col[top_indices]

        # Genres + tags
        genre_counts: Dict[str, int] = {}
        tag_counts: Dict[str, int] = {}
        for mid in top_movie_ids:
            info = metadata.get(mid)
            if info:
                for g in info['genres']:
                    if g != '(no genres listed)':
                        genre_counts[g] = genre_counts.get(g, 0) + 1
            for t in movie_top_tags.get(mid, []):
                tag_counts[t] = tag_counts.get(t, 0) + 1

        genres = sorted(genre_counts.keys(), key=lambda g: -genre_counts[g])[:5]
        tags = sorted(tag_counts.keys(), key=lambda t: -tag_counts[t])[:7]
        genre_summary = ', '.join(f"{g} ({genre_counts[g]})" for g in genres[:5])
        tag_summary = ', '.join(f"{t} ({tag_counts[t]})" for t in tags[:5])

        llm_result = label_neuron_with_llm(
            nid, top_movie_ids, top_weights, metadata, movie_top_tags,
            genre_summary=genre_summary, tag_summary=tag_summary,
        )

        if llm_result:
            label = llm_result['label']
            description = llm_result.get('description', f"More {label.lower()} movies")
            source = 'llm'
        else:
            parts = []
            if tags:
                parts.append(tags[0].title())
            if genres:
                parts.append(genres[0])
            label = ' · '.join(parts) if parts else f"Feature {nid}"
            description = f"More movies in the {label.lower()} space"
            source = 'metadata'

        new_labels[nid] = {
            'label': label, 'description': description,
            'label_source': source,
            'tags': tags[:5], 'genres': genres[:3],
            'activation_count': activation_count,
            'selectivity': round(selectivity, 4),
            'top_movies': top_movie_ids[:5],
        }

    # Update cache
    if new_labels:
        merged = {**existing, **new_labels}
        try:
            os.makedirs(data_dir, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump({str(k): v for k, v in merged.items()}, f, indent=1)
        except Exception:
            pass

    all_labels = {**existing, **new_labels}
    return {nid: all_labels[nid] for nid in neuron_ids if nid in all_labels}
