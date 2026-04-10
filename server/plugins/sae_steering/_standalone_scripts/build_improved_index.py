"""
Build improved direct text index for text steering.

This creates an index that maps text queries directly to SAE neurons
by using:
1. Neuron labels (from analyze_neurons.py)
2. Top movies for each neuron and their metadata
3. Tags/keywords associated with each neuron

This allows queries like "Tom Cruise action" to match neurons
that have Tom Cruise movies in their top activations.
"""

import json
import pickle
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR.parent.parent / "static" / "datasets" / "ml-latest"


def load_data():
    """Load neuron analysis and movie metadata."""
    
    # Neuron analysis
    with open(DATA_DIR / "neuron_analysis.json", 'r') as f:
        neuron_analysis = json.load(f)
    
    # Neuron labels
    with open(DATA_DIR / "neuron_labels.json", 'r') as f:
        neuron_labels = json.load(f)
    
    # TMDB data
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
    
    return neuron_analysis, neuron_labels, tmdb_data, movies


def create_neuron_description(neuron_id: str, analysis: dict, label: str, 
                               tmdb_data: dict, movies: dict) -> str:
    """
    Create a rich text description for a neuron based on what it represents.
    
    This combines:
    - The neuron label (e.g., "Thriller • 2000s")
    - Top genres
    - Top directors
    - Top actors
    - Keywords from top movies
    """
    parts = []
    
    # Add label
    if label:
        parts.append(label)
    
    # Get top genres
    genres = analysis.get('genres', {})
    if genres:
        top_genres = sorted(genres.items(), key=lambda x: -x[1])[:3]
        genre_str = ", ".join([g[0] for g in top_genres])
        parts.append(f"Genres: {genre_str}")
    
    # Get directors
    directors = analysis.get('directors', {})
    if directors:
        top_directors = sorted(directors.items(), key=lambda x: -x[1])[:3]
        director_str = ", ".join([d[0] for d in top_directors])
        parts.append(f"Directors: {director_str}")
    
    # Get cast
    cast = analysis.get('cast', {})
    if cast:
        top_cast = sorted(cast.items(), key=lambda x: -x[1])[:5]
        cast_str = ", ".join([c[0] for c in top_cast])
        parts.append(f"Starring: {cast_str}")
    
    # Get keywords
    keywords = analysis.get('keywords', {})
    if keywords:
        top_keywords = sorted(keywords.items(), key=lambda x: -x[1])[:5]
        keyword_str = ", ".join([k[0] for k in top_keywords])
        parts.append(f"Keywords: {keyword_str}")
    
    # Add some top movie titles
    top_movies = analysis.get('top_movies', [])
    if top_movies:
        titles = []
        for m in top_movies[:3]:
            movie_id = str(m.get('movie_id', ''))
            title = None
            
            # Try TMDB first
            if movie_id in tmdb_data:
                title = tmdb_data[movie_id].get('title')
            
            # Fall back to MovieLens
            if not title and movie_id in movies:
                title = movies[movie_id].get('title')
            
            if title:
                titles.append(title)
        
        if titles:
            parts.append(f"Movies: {', '.join(titles)}")
    
    return " | ".join(parts)


def build_improved_index():
    """Build the improved direct text index."""
    
    print("Loading data...")
    neuron_analysis, neuron_labels, tmdb_data, movies = load_data()
    
    print(f"Found {len(neuron_analysis)} neurons to index")
    
    # Load sentence transformer
    print("Loading sentence transformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    neuron_ids = []
    descriptions = []
    
    print("Creating neuron descriptions...")
    for neuron_id, analysis in tqdm(neuron_analysis.items()):
        if not neuron_id.isdigit():
            continue
            
        label = neuron_labels.get(neuron_id, f"Feature {neuron_id}")
        
        description = create_neuron_description(
            neuron_id, analysis, label, tmdb_data, movies
        )
        
        if description:
            neuron_ids.append(int(neuron_id))
            descriptions.append(description)
    
    print(f"Created {len(descriptions)} neuron descriptions")
    
    # Show some examples
    print("\nExample descriptions:")
    for i in range(min(3, len(descriptions))):
        print(f"  Neuron {neuron_ids[i]}: {descriptions[i][:100]}...")
    
    # Encode descriptions
    print("\nEncoding descriptions...")
    embeddings = model.encode(
        descriptions,
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=False
    )
    
    # Normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Save
    output_path = DATA_DIR / "direct_text_index.pt"
    torch.save({
        'neuron_ids': neuron_ids,
        'embedding_matrix': embeddings.astype(np.float32),
        'descriptions': descriptions,
        'version': 'v2_improved'
    }, output_path)
    
    print(f"\nSaved improved index to {output_path}")
    print(f"  {len(neuron_ids)} neurons indexed")
    print(f"  Embedding matrix shape: {embeddings.shape}")
    
    return neuron_ids, embeddings


def test_index():
    """Test the improved index with sample queries."""
    
    from text_steering import get_model, get_direct_index
    
    # Force reload
    import text_steering
    text_steering._direct_index = None
    
    model = get_model()
    index = get_direct_index()
    
    test_queries = [
        "Tom Cruise action movies",
        "romantic comedy",
        "Christopher Nolan sci-fi",
        "horror thriller",
        "animated children movies",
        "crime drama gangster",
    ]
    
    print("\nTesting queries:")
    print("=" * 60)
    
    for query in test_queries:
        query_emb = model.encode(query)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        
        similarities = index['embedding_matrix'] @ query_emb
        top_idx = np.argsort(similarities)[-5:][::-1]
        
        print(f"\n'{query}':")
        for idx in top_idx:
            neuron_id = index['neuron_ids'][idx]
            sim = similarities[idx]
            desc = index['descriptions'][idx][:60]
            print(f"  N{neuron_id} ({sim:.3f}): {desc}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test the index")
    args = parser.parse_args()
    
    if args.test:
        test_index()
    else:
        build_improved_index()
        test_index()
