#!/usr/bin/env python3
"""
Compute causal influence of each SAE neuron on recommendations.

This script measures how much each neuron actually affects the final
recommendation scores, not just how interpretable it is.

Metrics computed:
1. Decoder norm - how much neuron affects embedding reconstruction
2. Causal influence - how changing neuron affects prediction scores
3. Activation frequency - how often neuron fires across movies

Usage:
    python compute_neuron_influence.py

Output:
    data/neuron_influences.pt
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def load_models():
    """Load SAE and ELSA models."""
    import torch.nn as nn
    
    # Define TopKSAE class inline to avoid import issues
    class TopKSAE(nn.Module):
        def __init__(self, input_dim, feature_dim, k=32):
            super().__init__()
            self.input_dim = input_dim
            self.feature_dim = feature_dim
            self.k = k
            self.encoder = nn.Linear(input_dim, feature_dim, bias=True)
            self.decoder = nn.Linear(feature_dim, input_dim, bias=True)
        
        def encode(self, x):
            pre_activation = self.encoder(x)
            topk_values, topk_indices = torch.topk(pre_activation, self.k, dim=-1)
            activations = torch.zeros_like(pre_activation)
            activations.scatter_(-1, topk_indices, torch.relu(topk_values))
            return activations
        
        def decode(self, activations):
            return self.decoder(activations)
        
        def forward(self, x):
            return self.decode(self.encode(x))
    
    models_dir = Path(__file__).parent / "models"
    
    # Load SAE
    sae_path = models_dir / "prediction_aware_sae.pt"
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE model not found: {sae_path}")
    
    sae_data = torch.load(sae_path, map_location='cpu')
    config = sae_data['config']
    
    sae = TopKSAE(
        input_dim=config['input_dim'],
        feature_dim=config['hidden_dim'],
        k=config['k']
    )
    
    # Map state dict keys
    state_dict = sae_data['model_state_dict']
    new_state_dict = {}
    if 'encoder.weight' in state_dict:
        new_state_dict['encoder.weight'] = state_dict['encoder.weight']
    if 'decoder.encoder.weight' in state_dict:
        # Tied weights - decoder uses transposed encoder
        new_state_dict['decoder.weight'] = state_dict['decoder.encoder.weight'].T
    if 'pre_bias' in state_dict:
        new_state_dict['decoder.bias'] = state_dict['pre_bias']
    if 'latent_bias' in state_dict:
        new_state_dict['encoder.bias'] = state_dict['latent_bias']
    
    sae.load_state_dict(new_state_dict, strict=False)
    sae.eval()
    
    # Load ELSA
    elsa_path = models_dir / "elsa_model.pt"
    if elsa_path.exists():
        elsa_data = torch.load(elsa_path, map_location='cpu')
        print(f"Loaded ELSA model")
    else:
        elsa_data = None
        print("ELSA model not found, using decoder norm only")
    
    return sae, elsa_data


def compute_decoder_influence(sae):
    """
    Compute influence based on decoder weight norms.
    
    Neurons with larger decoder weights have more impact on reconstruction.
    """
    # decoder.weight shape: [input_dim, feature_dim]
    decoder_weights = sae.decoder.weight.detach()
    
    # L2 norm of each column = influence of each neuron
    influences = torch.norm(decoder_weights, dim=0).numpy()
    
    return influences


def compute_activation_stats(sae, movie_embeddings):
    """
    Compute activation statistics for each neuron.
    
    Returns:
        - mean_activation: average activation when neuron fires
        - activation_frequency: % of movies where neuron is active
    """
    with torch.no_grad():
        # Encode all movies
        activations = sae.encode(movie_embeddings)  # [num_movies, num_neurons]
    
    # Frequency: how often neuron is non-zero
    is_active = (activations > 0).float()
    frequency = is_active.mean(dim=0).numpy()
    
    # Mean activation when active
    sum_activations = activations.sum(dim=0)
    count_active = is_active.sum(dim=0).clamp(min=1)
    mean_when_active = (sum_activations / count_active).numpy()
    
    return {
        'frequency': frequency,
        'mean_activation': mean_when_active
    }


def compute_causal_influence(sae, movie_embeddings, elsa_weights=None, sample_size=500):
    """
    Measure causal effect of each neuron on predictions.
    
    For each neuron, we:
    1. Get baseline predictions
    2. Intervene by setting neuron to +1
    3. Measure change in predictions
    """
    num_neurons = sae.feature_dim
    num_movies = min(len(movie_embeddings), sample_size)
    
    # Sample movies for efficiency
    indices = np.random.choice(len(movie_embeddings), num_movies, replace=False)
    sample_embeddings = movie_embeddings[indices]
    
    influences = np.zeros(num_neurons)
    
    with torch.no_grad():
        # Baseline: encode -> decode
        baseline_activations = sae.encode(sample_embeddings)
        baseline_reconstructed = sae.decode(baseline_activations)
        
        # If we have ELSA weights, use them for prediction
        if elsa_weights is not None:
            baseline_scores = baseline_reconstructed @ elsa_weights.T
        else:
            baseline_scores = baseline_reconstructed
        
        # Test each neuron
        for neuron_idx in tqdm(range(num_neurons), desc="Computing causal influence"):
            # Intervene: add +1 to this neuron
            modified_activations = baseline_activations.clone()
            modified_activations[:, neuron_idx] += 1.0
            
            # Decode modified
            modified_reconstructed = sae.decode(modified_activations)
            
            if elsa_weights is not None:
                modified_scores = modified_reconstructed @ elsa_weights.T
            else:
                modified_scores = modified_reconstructed
            
            # Measure change
            delta = (modified_scores - baseline_scores).abs().mean().item()
            influences[neuron_idx] = delta
    
    return influences


def load_movie_embeddings():
    """Load pre-computed movie embeddings."""
    from plugins.utils.data_loading import load_ml_dataset
    
    models_dir = Path(__file__).parent / "models"
    embeddings_path = models_dir / "movie_embeddings.pt"
    
    if embeddings_path.exists():
        data = torch.load(embeddings_path, map_location='cpu')
        return data['embeddings']
    
    # Generate from ELSA if not cached
    print("Generating movie embeddings from ELSA...")
    
    elsa_path = models_dir / "elsa_model.pt"
    if not elsa_path.exists():
        raise FileNotFoundError("Need either movie_embeddings.pt or elsa_model.pt")
    
    elsa_data = torch.load(elsa_path, map_location='cpu')
    item_embeddings = elsa_data.get('item_embeddings')
    
    if item_embeddings is None:
        raise ValueError("ELSA model doesn't contain item_embeddings")
    
    return torch.tensor(item_embeddings, dtype=torch.float32)


def main():
    print("=" * 60)
    print("Computing SAE Neuron Influence Scores")
    print("=" * 60)
    
    # Load models
    print("\n1. Loading models...")
    sae, elsa_data = load_models()
    print(f"   SAE: {sae.input_dim} -> {sae.feature_dim} (k={sae.k})")
    
    # Load movie embeddings
    print("\n2. Loading movie embeddings...")
    try:
        movie_embeddings = load_movie_embeddings()
        print(f"   Loaded {len(movie_embeddings)} movie embeddings")
    except Exception as e:
        print(f"   Warning: Could not load embeddings: {e}")
        print("   Using random embeddings for decoder analysis only")
        movie_embeddings = torch.randn(1000, sae.input_dim)
    
    # Compute metrics
    results = {}
    
    # 1. Decoder influence (fast)
    print("\n3. Computing decoder influence...")
    decoder_influence = compute_decoder_influence(sae)
    results['decoder_influence'] = decoder_influence
    print(f"   Max: {decoder_influence.max():.4f}, Mean: {decoder_influence.mean():.4f}")
    
    # 2. Activation stats
    print("\n4. Computing activation statistics...")
    activation_stats = compute_activation_stats(sae, movie_embeddings)
    results['activation_frequency'] = activation_stats['frequency']
    results['mean_activation'] = activation_stats['mean_activation']
    active_neurons = (activation_stats['frequency'] > 0.01).sum()
    print(f"   Active neurons (>1% frequency): {active_neurons}")
    
    # 3. Causal influence (slower)
    print("\n5. Computing causal influence...")
    elsa_weights = None
    if elsa_data and 'item_embeddings' in elsa_data:
        elsa_weights = torch.tensor(elsa_data['item_embeddings'], dtype=torch.float32)
    
    causal_influence = compute_causal_influence(
        sae, movie_embeddings, elsa_weights, sample_size=300
    )
    results['causal_influence'] = causal_influence
    print(f"   Max: {causal_influence.max():.4f}, Mean: {causal_influence.mean():.4f}")
    
    # Compute combined score
    print("\n6. Computing combined influence score...")
    
    # Normalize each metric to [0, 1]
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)
    
    combined_score = (
        0.4 * normalize(causal_influence) +
        0.3 * normalize(decoder_influence) +
        0.2 * normalize(activation_stats['frequency']) +
        0.1 * normalize(activation_stats['mean_activation'])
    )
    results['combined_score'] = combined_score
    
    # Find top neurons
    top_indices = np.argsort(combined_score)[-20:][::-1]
    print(f"\n   Top 10 influential neurons:")
    for i, idx in enumerate(top_indices[:10]):
        print(f"   {i+1}. Neuron {idx}: score={combined_score[idx]:.3f}, "
              f"causal={causal_influence[idx]:.3f}, "
              f"freq={activation_stats['frequency'][idx]:.1%}")
    
    # Save results
    output_path = Path(__file__).parent / "data" / "neuron_influences.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'decoder_influence': decoder_influence,
        'causal_influence': causal_influence,
        'activation_frequency': activation_stats['frequency'],
        'mean_activation': activation_stats['mean_activation'],
        'combined_score': combined_score,
        'num_neurons': sae.feature_dim,
    }, output_path)
    
    print(f"\n✅ Saved neuron influences to {output_path}")
    
    return results


if __name__ == "__main__":
    main()
