"""
Test steering with both SAE models (basic vs prediction-aware).

Compares:
1. Basic SAE (sae_model_r4_k32.pt) - 135 active neurons
2. Prediction-aware SAE (prediction_aware_sae.pt) - 1576 active neurons

Tests natural language queries and compares recommendation changes.
"""

import sys
import os
import json
import csv
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR.parent.parent / "cache" / "utils" / "ml-latest"


# ========== ELSA Model (from train_elsa.py) ==========
class ELSA(nn.Module):
    """ELSA model - linear autoencoder for collaborative filtering"""
    def __init__(self, num_items, latent_dim):
        super().__init__()
        self.num_items = num_items
        self.latent_dim = latent_dim
        # Item embedding matrix
        self.A = nn.Parameter(torch.randn(num_items, latent_dim) * 0.01)
    
    def encode(self, x):
        """Encode user interactions to latent space: x @ A"""
        return torch.matmul(x, self.A)
    
    def decode(self, z):
        """Decode latent to scores: z @ A.T"""
        return torch.matmul(z, self.A.t())
    
    def forward(self, x):
        """Full pass: encode -> decode"""
        z = self.encode(x)
        return self.decode(z)
    
    def get_item_embeddings(self):
        return self.A.data


# ========== PredictionAwareSAE (from train_prediction_aware_sae.py) ==========
class TopKActivation(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
    
    def forward(self, x):
        topk = torch.topk(x, k=self.k, dim=-1)
        values = F.relu(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result


class TiedTranspose(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, x):
        return F.linear(x, self.encoder.weight.t())


class PredictionAwareSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, k, tied=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.activation = TopKActivation(k)
        
        if tied:
            self.decoder = TiedTranspose(self.encoder)
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        
        self.register_buffer('neuron_usage', torch.zeros(hidden_dim))
    
    def encode(self, x):
        x_centered = x - self.pre_bias
        pre_act = self.encoder(x_centered) + self.latent_bias
        return self.activation(pre_act)
    
    def forward(self, x):
        features = self.encode(x)
        x_hat = self.decoder(features) + self.pre_bias
        return x_hat, features


def load_elsa():
    """Load frozen ELSA model"""
    with open(DATA_DIR / "item2index.pkl", "rb") as f:
        item2index = pickle.load(f)
    
    num_items = len(item2index)
    latent_dim = 64  # from train_elsa.py
    
    elsa = ELSA(num_items, latent_dim)
    elsa.load_state_dict(torch.load(MODELS_DIR / "elsa_model_best.pt", map_location='cpu'))
    elsa.eval()
    
    return elsa, item2index


def load_sae(model_path: str):
    """Load SAE model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Infer dimensions from checkpoint
    if 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    else:
        state = checkpoint
    
    hidden_dim = state['encoder.weight'].shape[0]
    input_dim = state['encoder.weight'].shape[1]
    
    sae = PredictionAwareSAE(input_dim, hidden_dim, k=32)
    
    if 'model_state_dict' in checkpoint:
        sae.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        sae.load_state_dict(checkpoint, strict=False)
    
    sae.eval()
    return sae


def load_movie_metadata():
    """Load movie metadata for display"""
    movies = {}
    
    # MovieLens movies.csv
    if CACHE_DIR.exists():
        movies_path = CACHE_DIR / "movies.csv"
        if movies_path.exists():
            with open(movies_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    movie_id = int(row['movieId'])
                    movies[movie_id] = {
                        'title': row['title'],
                        'genres': row['genres'].split('|')
                    }
    
    # TMDB enrichment
    tmdb_path = DATA_DIR / "tmdb_data.json"
    if tmdb_path.exists():
        with open(tmdb_path, 'r', encoding='utf-8') as f:
            tmdb = json.load(f)
        for movie_id_str, data in tmdb.items():
            movie_id = int(movie_id_str)
            if movie_id in movies:
                movies[movie_id].update(data)
    
    return movies


def get_user_embedding(elsa, user_history: list, n_items: int = 83239):
    """Get user embedding from viewing history"""
    # Create user interaction vector
    user_vector = torch.zeros(n_items)
    for movie_idx in user_history:
        if 0 <= movie_idx < n_items:
            user_vector[movie_idx] = 1.0
    
    # Get user embedding through ELSA encoder
    with torch.no_grad():
        user_emb = elsa.encode(user_vector.unsqueeze(0))
    
    return user_emb.squeeze(0)


def get_recommendations(elsa, user_embedding, top_k: int = 20):
    """Get recommendations from user embedding"""
    with torch.no_grad():
        # Decode to get scores
        scores = elsa.decode(user_embedding.unsqueeze(0)).squeeze(0)
        
        # Get top-k
        top_scores, top_indices = torch.topk(scores, top_k)
        
    return list(zip(top_indices.numpy(), top_scores.numpy()))


def apply_sae_steering(sae, user_embedding, adjustments: dict, elsa, strength: float = 5.0):
    """
    Apply SAE-based steering using feature-weighted scoring.
    
    SAE is trained on ITEM embeddings, so we:
    1. Get item embeddings from ELSA
    2. Encode items to SAE features
    3. Re-weight scores based on feature adjustments
    
    Args:
        strength: How much to amplify steering effect (default 5.0)
    """
    with torch.no_grad():
        # Get all item embeddings
        item_embeddings = elsa.get_item_embeddings()  # (num_items, latent_dim)
        
        # Encode all items to SAE feature space
        item_features = sae.encode(item_embeddings)  # (num_items, hidden_dim)
        
        # Compute base scores (user_emb @ item_emb.T)
        base_scores = torch.matmul(user_embedding, item_embeddings.t())
        
        # Compute feature-based boost for each item
        feature_boost = torch.zeros(item_embeddings.shape[0])
        
        for neuron_id, adjustment in adjustments.items():
            if 0 <= neuron_id < item_features.shape[1]:
                # Items with high activation on this neuron get adjusted
                neuron_activation = item_features[:, neuron_id]
                # Normalize activation per neuron
                if neuron_activation.max() > 0:
                    neuron_activation = neuron_activation / neuron_activation.max()
                feature_boost += neuron_activation * adjustment * strength
        
        # Apply boost to scores (multiplicative for better effect)
        # Positive boost → increase score, Negative boost → decrease score
        score_multiplier = 1.0 + feature_boost.clamp(-0.9, 10.0)  # Limit negative to avoid zero
        modified_scores = base_scores * score_multiplier
        
    return modified_scores


def get_recommendations_from_scores(scores: torch.Tensor, top_k: int = 20):
    """Get top-k recommendations from scores tensor"""
    top_scores, top_indices = torch.topk(scores, top_k)
    return list(zip(top_indices.numpy(), top_scores.numpy()))


def text_to_adjustments_simple(text: str) -> dict:
    """
    Simple text to adjustment mapping.
    For testing - maps common preferences to neuron indices.
    """
    # Import the actual text steering
    try:
        from text_steering import text_to_adjustments
        return text_to_adjustments(text)
    except ImportError:
        # Fallback: return empty
        print("WARNING: text_steering not available, using empty adjustments")
        return {}


def compare_models(query: str, user_history: list):
    """Compare steering results between basic and prediction-aware SAE"""
    
    print(f"\n{'='*80}")
    print(f"Query: \"{query}\"")
    print(f"{'='*80}")
    
    # Load models
    elsa, item2index = load_elsa()
    index2item = {v: k for k, v in item2index.items()}  # Reverse mapping
    movies = load_movie_metadata()
    
    # Get user embedding
    user_emb = get_user_embedding(elsa, user_history)
    
    # Get text adjustments
    adjustments = text_to_adjustments_simple(query)
    
    if not adjustments:
        print("\nNo adjustments generated for this query.")
        print("This could mean the text steering index is not built yet.")
        return
    
    print(f"\nGenerated {len(adjustments)} neuron adjustments:")
    for neuron_id, adj in sorted(adjustments.items(), key=lambda x: -abs(x[1]))[:10]:
        sign = "+" if adj > 0 else ""
        print(f"  Neuron {neuron_id}: {sign}{adj:.3f}")
    
    def display_recs(recs, title, baseline_recs=None):
        """Display recommendations with movie info"""
        print(f"\n--- {title} ---")
        for idx, (item_idx, score) in enumerate(recs, 1):
            movie_id = index2item.get(int(item_idx), int(item_idx))
            if movie_id in movies:
                movie_title = movies[movie_id]['title']
                genres = movies[movie_id].get('genres', [])
                if baseline_recs:
                    baseline_rank = next((i+1 for i, (m, _) in enumerate(baseline_recs) if int(m) == int(item_idx)), None)
                    change = f" (was #{baseline_rank})" if baseline_rank else " (NEW)"
                else:
                    change = ""
                print(f"  {idx}. {movie_title} ({', '.join(genres[:3])}) - {score:.4f}{change}")
            else:
                print(f"  {idx}. Movie #{movie_id} - {score:.4f}")
    
    # Baseline recommendations
    baseline_recs = get_recommendations(elsa, user_emb, top_k=10)
    display_recs(baseline_recs, "BASELINE RECOMMENDATIONS")
    
    # Test with prediction-aware SAE
    pred_sae_path = MODELS_DIR / "prediction_aware_sae.pt"
    if pred_sae_path.exists():
        pred_sae = load_sae(str(pred_sae_path))
        steered_scores = apply_sae_steering(pred_sae, user_emb, adjustments, elsa)
        steered_recs = get_recommendations_from_scores(steered_scores, top_k=10)
        display_recs(steered_recs, "PREDICTION-AWARE SAE STEERING (1576 neurons)", baseline_recs)
    else:
        print(f"\nPrediction-aware SAE not found at {pred_sae_path}")
    
    # Test with basic SAE
    basic_sae_path = MODELS_DIR / "sae_model_r4_k32.pt"
    if basic_sae_path.exists():
        try:
            basic_sae = load_sae(str(basic_sae_path))
            steered_scores = apply_sae_steering(basic_sae, user_emb, adjustments, elsa)
            steered_recs = get_recommendations_from_scores(steered_scores, top_k=10)
            display_recs(steered_recs, "BASIC SAE STEERING (135 neurons)", baseline_recs)
        except Exception as e:
            print(f"  Could not load basic SAE: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nBasic SAE not found at {basic_sae_path}")


def main():
    """Run steering comparison tests"""
    
    # Sample user history (movie indices, not IDs)
    # Let's use some popular movies as history
    # These are example indices - ideally would load real user data
    sample_history = [0, 10, 50, 100, 200, 500, 1000, 2000, 5000]  # Example indices
    
    test_queries = [
        "I want more action movies",
        "No romantic comedies, more sci-fi",
        "Show me films like The Matrix",
        "Christopher Nolan movies",
        "More horror, less comedy",
        "Award-winning drama films",
        "Movies with Tom Hanks",
        "Animated family films",
    ]
    
    print("="*80)
    print("SAE STEERING COMPARISON TEST")
    print("="*80)
    print(f"\nUser history: {len(sample_history)} movies")
    
    for query in test_queries:
        try:
            compare_models(query, sample_history)
        except Exception as e:
            print(f"\nError testing query '{query}': {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
