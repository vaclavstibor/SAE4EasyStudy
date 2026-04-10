"""
Build Complete Text Steering Index

Combines all available data sources:
1. MovieLens genres & tags
2. Plot descriptions (descriptions.json)
3. TMDB cast & crew (if API key available)

Creates a searchable index for natural language movie queries.
"""

import os
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = Path("/Users/vaclav.stibor/Library/CloudStorage/OneDrive-HomeCreditInternationala.s/Documents/EasyStudy/server/static/datasets/ml-latest")

# Output files
TEXT_INDEX_FILE = DATA_DIR / "text_steering_index.pt"


def load_all_data():
    """Load all available data sources."""
    import pandas as pd
    
    print("Loading data sources...")
    
    # MovieLens movies
    movies_df = pd.read_csv(DATASET_DIR / "movies.csv")
    print(f"  Movies: {len(movies_df)}")
    
    # MovieLens tags
    tags_df = pd.read_csv(DATASET_DIR / "tags.csv")
    print(f"  Tags: {len(tags_df)}")
    
    # Links (TMDB/IMDB IDs)
    links_df = pd.read_csv(DATASET_DIR / "links.csv")
    movies_df = movies_df.merge(links_df[['movieId', 'tmdbId', 'imdbId']], on='movieId', how='left')
    
    # Plot descriptions
    desc_file = DATASET_DIR / "descriptions.json"
    descriptions = {}
    if desc_file.exists():
        with open(desc_file, 'r', encoding='utf-8') as f:
            desc_data = json.load(f)
            descriptions = {int(k): v.get("plot", "") for k, v in desc_data.items()}
        print(f"  Descriptions: {len(descriptions)}")
    
    # Item2index mapping
    with open(DATA_DIR / "item2index.pkl", "rb") as f:
        item2index = pickle.load(f)
    print(f"  Items in SAE index: {len(item2index)}")
    
    # TMDB cache (from enrichment or direct API)
    tmdb_cache = {}
    # Try multiple locations
    tmdb_locations = [
        DATASET_DIR / "tmdb_data.json",  # From enrich_from_kaggle.py
        DATA_DIR / "tmdb_cache.json",    # From old API download
    ]
    for tmdb_file in tmdb_locations:
        if tmdb_file.exists():
            with open(tmdb_file, 'r', encoding='utf-8') as f:
                tmdb_cache = json.load(f)
            print(f"  TMDB cache: {len(tmdb_cache)} (from {tmdb_file.name})")
            break
    
    # Enriched tags (from enrich_from_kaggle.py)
    enriched_tags = {}
    enriched_file = DATA_DIR / "enriched_tags.pkl"
    if enriched_file.exists():
        with open(enriched_file, 'rb') as f:
            enriched_tags = pickle.load(f)
        print(f"  Enriched tags: {len(enriched_tags)}")
    
    return movies_df, tags_df, descriptions, item2index, tmdb_cache, enriched_tags


def build_movie_documents(movies_df, tags_df, descriptions, item2index, tmdb_cache, enriched_tags) -> Dict[int, str]:
    """
    Build text documents for each movie combining all available info.
    
    Each document contains:
    - Title
    - Genres
    - Plot description
    - User tags
    - Cast & crew (if available)
    """
    from collections import Counter
    
    print("\nBuilding movie documents...")
    
    # Group tags by movie
    tags_df['tag'] = tags_df['tag'].str.lower()
    movie_tags = tags_df.groupby('movieId')['tag'].apply(list).to_dict()
    
    documents = {}
    
    for _, row in movies_df.iterrows():
        movie_id = row['movieId']
        
        if movie_id not in item2index:
            continue
        
        item_idx = item2index[movie_id]
        doc_parts = []
        
        # Title
        title = row['title']
        doc_parts.append(f"title: {title}")
        
        # Genres
        genres = row.get('genres', '')
        if genres and genres != "(no genres listed)":
            genre_list = [g.strip().lower() for g in genres.split('|')]
            doc_parts.append(f"genres: {' '.join(genre_list)}")
        
        # Plot description
        if movie_id in descriptions and descriptions[movie_id]:
            doc_parts.append(f"plot: {descriptions[movie_id]}")
        
        # User tags (most common)
        if movie_id in movie_tags:
            tags = movie_tags[movie_id]
            tag_counts = Counter(tags)
            top_tags = [t for t, _ in tag_counts.most_common(10)]
            doc_parts.append(f"tags: {' '.join(top_tags)}")
        
        # TMDB data (cast, director)
        tmdb_id = row.get('tmdbId')
        if tmdb_id and not np.isnan(tmdb_id):
            tmdb_data = tmdb_cache.get(str(int(tmdb_id)))
            if tmdb_data:
                # Cast
                cast_names = [a['name'].lower() for a in tmdb_data.get('cast', [])[:5]]
                if cast_names:
                    doc_parts.append(f"actors: {' '.join(cast_names)}")
                
                # Directors
                directors = [d.lower() for d in tmdb_data.get('directors', [])]
                if directors:
                    doc_parts.append(f"director: {' '.join(directors)}")
                
                # Keywords
                keywords = [k.lower() for k in tmdb_data.get('keywords', [])[:10]]
                if keywords:
                    doc_parts.append(f"keywords: {' '.join(keywords)}")
        
        documents[item_idx] = " | ".join(doc_parts)
    
    print(f"  Created {len(documents)} movie documents")
    return documents


def build_tag_index(movies_df, tags_df, descriptions, item2index, tmdb_cache, enriched_tags) -> Dict[str, List[int]]:
    """
    Build searchable tag index from all data sources.
    
    Tags are prefixed by type:
    - genre:action
    - actor:tom cruise
    - director:christopher nolan
    - tag:twist ending
    - keyword:time travel
    - decade:1990s
    """
    print("\nBuilding tag index...")
    
    tag_index = defaultdict(list)
    
    for _, row in movies_df.iterrows():
        movie_id = row['movieId']
        
        if movie_id not in item2index:
            continue
        
        item_idx = item2index[movie_id]
        
        # Genres
        genres = row.get('genres', '')
        if genres and genres != "(no genres listed)":
            for genre in genres.split('|'):
                tag_index[f"genre:{genre.strip().lower()}"].append(item_idx)
        
        # TMDB data
        tmdb_id = row.get('tmdbId')
        if tmdb_id and not np.isnan(tmdb_id):
            tmdb_data = tmdb_cache.get(str(int(tmdb_id)))
            if tmdb_data:
                # Actors
                for i, actor in enumerate(tmdb_data.get('cast', [])[:10]):
                    name = actor['name'].lower()
                    tag_index[f"actor:{name}"].append(item_idx)
                    if i < 3:  # Lead actors
                        tag_index[f"lead:{name}"].append(item_idx)
                
                # Directors
                for director in tmdb_data.get('directors', []):
                    tag_index[f"director:{director.lower()}"].append(item_idx)
                
                # Keywords
                for keyword in tmdb_data.get('keywords', []):
                    tag_index[f"keyword:{keyword.lower()}"].append(item_idx)
                
                # Decade
                year = tmdb_data.get('release_year', '')
                if year and str(year).isdigit():
                    decade = (int(year) // 10) * 10
                    tag_index[f"decade:{decade}s"].append(item_idx)
    
    # Add MovieLens user tags
    tags_df['tag'] = tags_df['tag'].str.lower()
    for _, row in tags_df.iterrows():
        movie_id = row['movieId']
        tag = row['tag']
        
        if movie_id in item2index and len(tag) >= 2:
            tag_index[f"tag:{tag}"].append(item2index[movie_id])
    
    # Filter by minimum occurrences
    min_count = 3
    filtered_index = {
        tag: list(set(items))  # Remove duplicates
        for tag, items in tag_index.items()
        if len(set(items)) >= min_count
    }
    
    # Stats
    categories = defaultdict(int)
    for tag in filtered_index:
        cat = tag.split(':')[0]
        categories[cat] += 1
    
    print(f"  Total tags: {len(filtered_index)}")
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count}")
    
    return filtered_index


def compute_tag_neuron_mapping(tag_index: Dict[str, List[int]], num_items: int, 
                                use_prediction_aware: bool = False) -> Tuple[torch.Tensor, List[str]]:
    """
    Compute tag-to-neuron mapping using SAE activations.
    
    For each tag, computes the average SAE activation pattern
    of all movies with that tag.
    
    Args:
        tag_index: Dict mapping tags to item indices
        num_items: Total number of items
        use_prediction_aware: If True, use prediction_aware_sae.pt, else sae_model_r4_k32.pt
    """
    from train_elsa import ELSA, latent_dim
    
    print("\nComputing tag-neuron mapping...")
    print(f"  Using {'prediction-aware' if use_prediction_aware else 'basic'} SAE")
    
    # Load ELSA
    elsa = ELSA(num_items, latent_dim)
    elsa.load_state_dict(torch.load(MODELS_DIR / "elsa_model_best.pt", map_location='cpu'))
    elsa.eval()
    
    # Load SAE based on choice
    if use_prediction_aware:
        from train_prediction_aware_sae import PredictionAwareSAE
        checkpoint = torch.load(MODELS_DIR / "prediction_aware_sae.pt", map_location='cpu')
        config = checkpoint['config']
        sae = PredictionAwareSAE(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            k=config['k'],
            tied=config.get('tied', True)
        )
        sae.load_state_dict(checkpoint['model_state_dict'], strict=False)
        hidden_dim = config['hidden_dim']
    else:
        from train_sae import TopKSAE, k, hidden_dim
        sae = TopKSAE(latent_dim, hidden_dim, k)
        sae.load_state_dict(torch.load(MODELS_DIR / "sae_model_r4_k32.pt", map_location='cpu'))
    
    sae.eval()
    
    # Compute sparse embeddings for all items
    print("  Computing SAE activations...")
    with torch.no_grad():
        embeddings = elsa.A.clone()
        embeddings_norm = torch.nn.functional.normalize(embeddings, dim=1)
        
        if use_prediction_aware:
            h_sparse = sae.encode(embeddings_norm)
        else:
            _, h_sparse, _ = sae(embeddings_norm)
    
    print(f"  Sparse embeddings shape: {h_sparse.shape}")
    
    # Check active neurons
    active_per_neuron = (h_sparse > 0).sum(dim=0)
    active_neurons = (active_per_neuron > 0).sum().item()
    print(f"  Active neurons: {active_neurons}/{hidden_dim}")
    
    # Compute tag vectors using RATIO-based discriminative method
    # For each tag, we compute: tag_mean / global_mean
    # This gives us neurons that fire MORE for this tag relative to all items
    print("  Computing discriminative tag vectors (ratio method)...")
    tag_names = list(tag_index.keys())
    
    # Global mean activation per neuron (across all items)
    global_mean = h_sparse.mean(dim=0) + 1e-6  # Shape: [hidden_dim], add epsilon to avoid div by zero
    
    tag_vectors = []
    
    for tag in tag_names:
        items = tag_index[tag]
        if items and len(items) >= 3:
            tag_activations = h_sparse[items]  # Shape: [num_items_with_tag, hidden_dim]
            
            # Compute mean activation for items with this tag
            tag_mean = tag_activations.mean(dim=0) + 1e-6
            
            # Compute ratio: how much MORE does each neuron fire for this tag vs globally?
            # ratio > 1.0 means neuron is discriminative for this tag
            ratio = tag_mean / global_mean
            
            # Convert to discriminative score:
            # - ratio = 1.0 -> score = 0 (same as global, not discriminative)
            # - ratio = 2.0 -> score = 1.0 (2x more active, very discriminative)
            # - ratio = 0.5 -> score = -1.0 (2x less active, negatively discriminative)
            # Using: score = ratio - 1 (clamped to positive)
            discriminative_score = torch.relu(ratio - 1.0)
            
            # Normalize to unit vector
            if discriminative_score.norm() > 0:
                discriminative_score = torch.nn.functional.normalize(discriminative_score, dim=0)
            
            tag_vectors.append(discriminative_score)
        else:
            tag_vectors.append(torch.zeros(hidden_dim))
    
    tag_tensor = torch.stack(tag_vectors)
    print(f"  Tag tensor shape: {tag_tensor.shape}")
    
    # Verify the ratio-based approach worked
    genre_tags = [i for i, t in enumerate(tag_names) if t.startswith('genre:')]
    if genre_tags:
        print(f"  Sample genre tag neurons (top 5):")
        for idx in genre_tags[:8]:
            top5 = torch.topk(tag_tensor[idx], 5)
            neurons = top5.indices.tolist()
            vals = [f"{v:.3f}" for v in top5.values.tolist()]
            print(f"    {tag_names[idx]}: {neurons} {vals}")
    
    return tag_tensor, tag_names


def build_sentence_embeddings(tag_names: List[str]) -> np.ndarray:
    """
    Compute Sentence-BERT embeddings for all tag names.
    
    This allows matching user queries to tags using semantic similarity.
    """
    from sentence_transformers import SentenceTransformer
    
    print("\nComputing Sentence-BERT embeddings for tags...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Clean tag names for embedding
    clean_names = []
    for tag in tag_names:
        # Remove prefix (genre:, actor:, etc.)
        if ':' in tag:
            name = tag.split(':', 1)[1]
        else:
            name = tag
        clean_names.append(name)
    
    embeddings = model.encode(clean_names, show_progress_bar=True)
    print(f"  Embeddings shape: {embeddings.shape}")
    
    return embeddings


def main(use_prediction_aware: bool = False):
    """Build complete text steering index."""
    print("=" * 60)
    print("BUILDING TEXT STEERING INDEX")
    print(f"SAE Model: {'prediction-aware' if use_prediction_aware else 'basic'}")
    print("=" * 60)
    
    # Load all data
    movies_df, tags_df, descriptions, item2index, tmdb_cache, enriched_tags = load_all_data()
    num_items = len(item2index)
    
    # Use enriched_tags directly if available (much simpler!)
    if enriched_tags:
        print("\nUsing pre-computed enriched tags...")
        tag_index = enriched_tags
    else:
        # Build tag index from scratch
        tag_index = build_tag_index(movies_df, tags_df, descriptions, item2index, tmdb_cache, enriched_tags)
    
    # Compute tag-neuron mapping
    tag_tensor, tag_names = compute_tag_neuron_mapping(tag_index, num_items, use_prediction_aware)
    
    # Compute sentence embeddings for tags
    tag_embeddings = build_sentence_embeddings(tag_names)
    
    # Save everything
    print("\n" + "=" * 60)
    print("SAVING INDEX")
    print("=" * 60)
    
    DATA_DIR.mkdir(exist_ok=True)
    
    index_data = {
        "tag_names": tag_names,
        "tag_tensor": tag_tensor,  # [num_tags x num_neurons]
        "tag_embeddings": torch.from_numpy(tag_embeddings),  # [num_tags x embedding_dim]
        "tag_index": tag_index,  # tag -> list of item indices
        "num_items": num_items,
        "num_tags": len(tag_names),
        "num_neurons": tag_tensor.shape[1],
    }
    
    torch.save(index_data, TEXT_INDEX_FILE)
    print(f"Saved: {TEXT_INDEX_FILE}")
    print(f"  Tags: {len(tag_names)}")
    print(f"  Neurons: {tag_tensor.shape[1]}")
    
    # Show sample queries
    print("\n" + "=" * 60)
    print("SAMPLE TAG QUERIES")
    print("=" * 60)
    
    sample_queries = [
        "tom cruise",
        "action",
        "christopher nolan",
        "time travel",
        "1990s",
    ]
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for query in sample_queries:
        query_emb = model.encode(query)
        
        # Compute similarities
        similarities = np.dot(tag_embeddings, query_emb) / (
            np.linalg.norm(tag_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        
        top_indices = np.argsort(similarities)[-5:][::-1]
        
        print(f"\nQuery: '{query}'")
        for idx in top_indices:
            print(f"  {tag_names[idx]}: {similarities[idx]:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-aware", action="store_true", 
                        help="Use prediction-aware SAE instead of basic SAE")
    args = parser.parse_args()
    
    main(use_prediction_aware=args.prediction_aware)
