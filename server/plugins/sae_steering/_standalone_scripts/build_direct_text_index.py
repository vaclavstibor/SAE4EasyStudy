"""
Build Direct Text-to-Neuron Index

Creates a more direct mapping from text queries to SAE neurons by:
1. Computing text embeddings for all movies using their metadata
2. Finding which neurons activate most for movies with similar text embeddings

This improves upon tag-based steering by using the full semantic content of movies.

Usage:
    python build_direct_text_index.py
"""

import json
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR.parent.parent / "static" / "datasets" / "ml-latest"


def load_all_data():
    """Load all necessary data."""
    print("Loading data...")
    
    # Item mappings
    with open(DATA_DIR / "item2index.pkl", "rb") as f:
        item2index = pickle.load(f)
    index2item = {v: k for k, v in item2index.items()}
    
    # ELSA embeddings
    from train_elsa import ELSA, latent_dim
    elsa = ELSA(len(item2index), latent_dim)
    elsa.load_state_dict(torch.load(MODELS_DIR / "elsa_model_best.pt", map_location='cpu'))
    elsa.eval()
    elsa_embeddings = F.normalize(elsa.A.detach(), dim=1)
    
    # SAE model
    from train_sae import TopKSAE, hidden_dim, k
    sae = TopKSAE(latent_dim, hidden_dim, k)
    sae.load_state_dict(torch.load(MODELS_DIR / "sae_model_r4_k32.pt", map_location='cpu'))
    sae.eval()
    
    # TMDB metadata
    tmdb_path = DATASET_DIR / "tmdb_data.json"
    if tmdb_path.exists():
        with open(tmdb_path, 'r', encoding='utf-8') as f:
            tmdb_data = json.load(f)
    else:
        tmdb_data = {}
    
    # MovieLens movies
    movies_path = DATASET_DIR / "movies.csv"
    movies = {}
    with open(movies_path, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                movie_id = parts[0]
                if len(parts) > 3:
                    title = ','.join(parts[1:-1]).strip('"')
                    genres = parts[-1]
                else:
                    title = parts[1].strip('"')
                    genres = parts[2]
                movies[movie_id] = {'title': title, 'genres': genres.split('|')}
    
    # Get SAE activations
    print("Computing SAE activations...")
    with torch.no_grad():
        _, activations, _ = sae(elsa_embeddings)
    
    return elsa_embeddings, activations, item2index, index2item, tmdb_data, movies


def create_movie_text(movie_id: str, tmdb_data: dict, movies: dict) -> str:
    """Create rich text description for a movie."""
    tmdb_info = tmdb_data.get(movie_id, {})
    ml_info = movies.get(movie_id, {})
    
    parts = []
    
    # Title
    title = tmdb_info.get('title') or ml_info.get('title', '')
    if title:
        parts.append(title)
    
    # Genres
    genres = tmdb_info.get('genres') or ml_info.get('genres', [])
    if isinstance(genres, str):
        genres = genres.split('|')
    if genres:
        parts.append(f"Genres: {', '.join(genres)}")
    
    # Directors
    directors = tmdb_info.get('directors', [])
    if directors:
        parts.append(f"Director: {', '.join(directors[:2])}")
    
    # Cast
    cast = tmdb_info.get('cast', [])
    if cast:
        actor_names = [a.get('name') if isinstance(a, dict) else a for a in cast[:5]]
        parts.append(f"Cast: {', '.join(actor_names)}")
    
    # Keywords
    keywords = tmdb_info.get('keywords', [])
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords[:10])}")
    
    # Overview
    overview = tmdb_info.get('overview', '')
    if overview:
        parts.append(overview[:300])
    
    return ' | '.join(parts) if parts else title


def build_neuron_concept_index():
    """
    Build an index that maps text concepts to neurons.
    
    For each selective neuron, we find the most descriptive text by:
    1. Getting movies that strongly activate the neuron
    2. Extracting common text patterns from those movies
    3. Creating a "concept embedding" for each neuron
    """
    from sentence_transformers import SentenceTransformer
    
    elsa_embeddings, activations, item2index, index2item, tmdb_data, movies = load_all_data()
    
    # Load neuron analysis to get selective neurons
    analysis_file = DATA_DIR / "neuron_analysis.json"
    with open(analysis_file, 'r', encoding='utf-8') as f:
        neuron_analysis = json.load(f)
    
    selective_neurons = [
        int(neuron_id) for neuron_id, info in neuron_analysis.items()
        if info.get('selective', False)
    ]
    print(f"Found {len(selective_neurons)} selective neurons")
    
    # Initialize text encoder
    print("Loading Sentence-BERT model...")
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    # For each selective neuron, create a concept embedding
    concept_embeddings = {}
    concept_texts = {}
    
    print("Building concept embeddings for each neuron...")
    for neuron_idx in tqdm(selective_neurons):
        # Get top activating movies for this neuron
        neuron_acts = activations[:, neuron_idx]
        top_k = min(50, (neuron_acts > 0).sum().item())
        if top_k == 0:
            continue
        
        top_indices = torch.topk(neuron_acts, top_k).indices.tolist()
        
        # Collect text descriptions
        texts = []
        for idx in top_indices[:20]:  # Use top 20 for concept
            movie_id = str(index2item[idx])
            text = create_movie_text(movie_id, tmdb_data, movies)
            if text:
                texts.append(text)
        
        if not texts:
            continue
        
        # Compute average embedding (concept centroid)
        text_embeddings = sbert.encode(texts, convert_to_tensor=True)
        concept_centroid = text_embeddings.mean(dim=0).cpu()
        
        concept_embeddings[neuron_idx] = concept_centroid.numpy()
        
        # Store sample texts for debugging
        concept_texts[neuron_idx] = texts[:3]
    
    print(f"Created concept embeddings for {len(concept_embeddings)} neurons")
    
    # Stack into matrix for efficient similarity search
    neuron_ids = list(concept_embeddings.keys())
    embedding_matrix = np.stack([concept_embeddings[n] for n in neuron_ids])
    
    # Normalize for cosine similarity
    embedding_matrix = embedding_matrix / (np.linalg.norm(embedding_matrix, axis=1, keepdims=True) + 1e-8)
    
    # Save index
    index_data = {
        'neuron_ids': neuron_ids,
        'embedding_matrix': torch.tensor(embedding_matrix),
        'concept_texts': concept_texts,  # For debugging
        'sbert_model': 'all-MiniLM-L6-v2'
    }
    
    output_path = DATA_DIR / "direct_text_index.pt"
    torch.save(index_data, output_path)
    print(f"Saved to: {output_path}")
    
    # Print sample concepts
    print("\nSample neuron concepts:")
    for neuron_idx in neuron_ids[:10]:
        label = neuron_analysis[str(neuron_idx)].get('label', 'Unknown')
        sample_movies = concept_texts.get(neuron_idx, [])[:2]
        print(f"\nNeuron {neuron_idx} ({label}):")
        for movie in sample_movies:
            # Print just first 100 chars
            print(f"  - {movie[:100]}...")
    
    return index_data


def test_direct_index():
    """Test the direct text index with sample queries."""
    from sentence_transformers import SentenceTransformer
    
    # Load index
    index_data = torch.load(DATA_DIR / "direct_text_index.pt", map_location='cpu')
    neuron_ids = index_data['neuron_ids']
    embedding_matrix = index_data['embedding_matrix'].numpy()
    
    # Load labels
    with open(DATA_DIR / "neuron_labels.json", 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    test_queries = [
        "Christopher Nolan sci-fi",
        "romantic comedy",
        "horror movies",
        "action movies with Tom Cruise",
        "Quentin Tarantino films",
        "animated movies for kids",
        "psychological thriller",
        "classic western",
    ]
    
    print("\n" + "="*80)
    print("TESTING DIRECT TEXT INDEX")
    print("="*80)
    
    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        
        # Encode query
        query_emb = sbert.encode(query)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        
        # Find most similar neurons
        similarities = embedding_matrix @ query_emb
        top_indices = np.argsort(similarities)[-5:][::-1]
        
        print("Top matching neurons:")
        for idx in top_indices:
            neuron_id = neuron_ids[idx]
            sim = similarities[idx]
            label = labels.get(str(neuron_id), f'Neuron {neuron_id}')
            print(f"  {neuron_id}: {sim:.3f} - {label}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_direct_index()
    else:
        build_neuron_concept_index()
        test_direct_index()
