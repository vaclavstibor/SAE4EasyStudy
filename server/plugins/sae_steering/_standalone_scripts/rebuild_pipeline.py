#!/usr/bin/env python
"""
Complete Pipeline Rebuild for SAE Steering

This script rebuilds all components needed for SAE steering:
1. Train prediction-aware SAE (or use existing)
2. Generate item SAE features cache
3. Build text steering index
4. Generate neuron analysis and labels

Usage:
    python rebuild_pipeline.py                    # Full rebuild
    python rebuild_pipeline.py --skip-training    # Skip SAE training
    python rebuild_pipeline.py --quick            # Quick test (fewer epochs)
"""

import argparse
import os
import sys
import json
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR.parent.parent / "static" / "datasets" / "ml-latest"


def step1_train_sae(quick=False):
    """Train prediction-aware SAE."""
    print("\n" + "="*60)
    print("STEP 1: Training Prediction-Aware SAE")
    print("="*60)
    
    from train_prediction_aware_sae import train_prediction_aware_sae, EPOCHS
    
    if quick:
        # Temporarily reduce epochs for quick testing
        import train_prediction_aware_sae as trainer
        original_epochs = trainer.EPOCHS
        trainer.EPOCHS = 50
        print(f"Quick mode: {trainer.EPOCHS} epochs")
    
    sae = train_prediction_aware_sae()
    
    if quick:
        trainer.EPOCHS = original_epochs
    
    return sae


def step2_generate_features():
    """Generate SAE features for all items."""
    print("\n" + "="*60)
    print("STEP 2: Generating Item SAE Features")
    print("="*60)
    
    from train_prediction_aware_sae import PredictionAwareSAE
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load SAE
    checkpoint = torch.load(MODELS_DIR / "prediction_aware_sae.pt", map_location=device)
    config = checkpoint['config']
    
    sae = PredictionAwareSAE(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        k=config['k'],
        tied=config.get('tied', True)
    ).to(device)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()
    
    print(f"Loaded SAE: {config}")
    
    # Load ELSA embeddings
    from train_elsa import ELSA, latent_dim
    
    with open(DATA_DIR / "item2index.pkl", "rb") as f:
        item2index = pickle.load(f)
    
    num_items = len(item2index)
    index2item = {v: k for k, v in item2index.items()}
    
    elsa = ELSA(num_items, latent_dim)
    elsa.load_state_dict(torch.load(MODELS_DIR / "elsa_model_best.pt", map_location=device))
    elsa.eval()
    
    # Get normalized embeddings
    with torch.no_grad():
        item_embeddings = F.normalize(elsa.A.detach(), dim=1)
    
    print(f"Item embeddings: {item_embeddings.shape}")
    
    # Save item_embeddings.pt (needed by sae_recommender)
    item_ids = [index2item[i] for i in range(num_items)]
    torch.save({
        'embeddings': item_embeddings.cpu(),
        'item_ids': item_ids
    }, DATA_DIR / "item_embeddings.pt")
    print(f"Saved item_embeddings.pt")
    
    # Compute SAE features
    with torch.no_grad():
        features = sae.get_feature_activations(item_embeddings.to(device))
    
    print(f"SAE features: {features.shape}")
    
    # Save features
    torch.save({
        'features': features.cpu(),
        'item_ids': item_ids
    }, DATA_DIR / "item_sae_features_prediction_aware.pt")
    
    print(f"Saved item_sae_features_prediction_aware.pt")
    
    # Stats
    active_neurons = (features > 0).any(dim=0).sum().item()
    avg_active = (features > 0).sum(dim=1).float().mean().item()
    print(f"Active neurons: {active_neurons}/{config['hidden_dim']}")
    print(f"Avg active per item: {avg_active:.1f}")
    
    return features.cpu(), item_ids


def step3_analyze_neurons(features, item_ids):
    """Analyze neuron selectivity and generate labels."""
    print("\n" + "="*60)
    print("STEP 3: Analyzing Neurons & Generating Labels")
    print("="*60)
    
    import pandas as pd
    
    # Load movie metadata
    movies_df = pd.read_csv(DATASET_DIR / "movies.csv")
    movies = {row['movieId']: {'title': row['title'], 'genres': row['genres'].split('|')} 
              for _, row in movies_df.iterrows()}
    
    # Create movie_id to index mapping
    id_to_idx = {mid: i for i, mid in enumerate(item_ids)}
    
    # Analyze each genre
    genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    # Find movies per genre
    genre_movies = defaultdict(list)
    for movie_id, info in movies.items():
        if movie_id in id_to_idx:
            idx = id_to_idx[movie_id]
            for genre in info['genres']:
                if genre in genres:
                    genre_movies[genre].append(idx)
    
    print(f"Movies per genre:")
    for genre, idxs in sorted(genre_movies.items(), key=lambda x: -len(x[1])):
        print(f"  {genre}: {len(idxs)}")
    
    # Find best neurons for each genre
    genre_neurons = {}
    neuron_labels = {}
    neuron_analysis = {}
    
    num_neurons = features.shape[1]
    
    for genre, movie_idxs in genre_movies.items():
        if len(movie_idxs) < 50:
            continue
            
        genre_features = features[movie_idxs]  # (N_genre, neurons)
        other_idxs = [i for i in range(len(item_ids)) if i not in movie_idxs]
        other_features = features[other_idxs]
        
        # Average activation in-genre vs out-of-genre
        in_avg = genre_features.mean(dim=0)
        out_avg = other_features.mean(dim=0)
        
        # Specificity = in_avg / (out_avg + epsilon)
        specificity = in_avg / (out_avg + 0.001)
        
        # Find top neurons for this genre
        top_neurons = torch.topk(specificity, 5).indices.tolist()
        genre_neurons[genre] = {
            'top_neurons': top_neurons,
            'specificity': [specificity[n].item() for n in top_neurons]
        }
        
        # Label the most specific neuron
        best_neuron = top_neurons[0]
        if best_neuron not in neuron_labels or specificity[best_neuron] > neuron_analysis.get(best_neuron, {}).get('specificity', 0):
            neuron_labels[best_neuron] = genre
            neuron_analysis[best_neuron] = {
                'label': genre,
                'specificity': specificity[best_neuron].item(),
                'in_genre_activation': in_avg[best_neuron].item(),
                'out_genre_activation': out_avg[best_neuron].item(),
                'selective': True
            }
    
    print(f"\nGenre-specific neurons found:")
    for genre, info in genre_neurons.items():
        print(f"  {genre}: Neurons {info['top_neurons'][:3]}, spec={info['specificity'][0]:.2f}")
    
    # Find all selective neurons (those with high variance in activation)
    activation_rate = (features > 0).float().mean(dim=0)
    for neuron_idx in range(num_neurons):
        rate = activation_rate[neuron_idx].item()
        if 0.05 < rate < 0.3:  # Selective: activates on 5-30% of movies
            if neuron_idx not in neuron_analysis:
                neuron_analysis[neuron_idx] = {
                    'label': f'Feature {neuron_idx}',
                    'activation_rate': rate,
                    'selective': True
                }
    
    # Save
    with open(DATA_DIR / "neuron_labels.json", 'w') as f:
        json.dump({str(k): v for k, v in neuron_labels.items()}, f, indent=2)
    
    with open(DATA_DIR / "neuron_analysis.json", 'w') as f:
        json.dump({str(k): v for k, v in neuron_analysis.items()}, f, indent=2)
    
    with open(DATA_DIR / "genre_neurons_prediction_aware.json", 'w') as f:
        json.dump(genre_neurons, f, indent=2)
    
    print(f"\nSaved neuron_labels.json ({len(neuron_labels)} labels)")
    print(f"Saved neuron_analysis.json ({len(neuron_analysis)} selective neurons)")
    
    return genre_neurons, neuron_analysis


def step4_build_text_index():
    """Build text steering index."""
    print("\n" + "="*60)
    print("STEP 4: Building Text Steering Index")
    print("="*60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "build_text_index.py", "--prediction-aware"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def step5_test_steering():
    """Test the steering pipeline."""
    print("\n" + "="*60)
    print("STEP 5: Testing Steering Pipeline")
    print("="*60)
    
    import csv
    
    # Reload modules
    import importlib
    import text_steering
    text_steering._index_data = None
    importlib.reload(text_steering)
    
    from text_steering import text_to_adjustments
    from sae_recommender import SAERecommender
    
    # Load movie titles
    movies = {}
    with open(DATASET_DIR / "movies.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies[int(row['movieId'])] = {'title': row['title'], 'genres': row['genres'].split('|')}
    
    # Test queries
    test_queries = [
        "I like action movies, no romance",
        "More sci-fi and fantasy",
        "Classic drama films",
    ]
    
    rec = SAERecommender()
    
    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")
        
        adjustments = text_to_adjustments(query, sensitivity=3.0)
        
        if not adjustments:
            print("  No adjustments generated!")
            continue
        
        print(f"  Adjustments: {len(adjustments)} neurons")
        
        recs = rec.get_recommendations(adjustments, n_items=10)
        
        action_count = sum(1 for r in recs if 'Action' in movies.get(r['movie_id'], {}).get('genres', []))
        romance_count = sum(1 for r in recs if 'Romance' in movies.get(r['movie_id'], {}).get('genres', []))
        
        print(f"  Results: {action_count} action, {romance_count} romance")
        for i, r in enumerate(recs[:5]):
            info = movies.get(r['movie_id'], {'title': 'Unknown', 'genres': []})
            genres = '|'.join(info['genres'][:3])
            print(f"    {i+1}. {info['title'][:40]} [{genres}] - {r['score']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Rebuild SAE Steering Pipeline")
    parser.add_argument("--skip-training", action="store_true", help="Skip SAE training")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer epochs")
    parser.add_argument("--test-only", action="store_true", help="Only run final test")
    args = parser.parse_args()
    
    print("="*60)
    print("SAE STEERING PIPELINE REBUILD")
    print("="*60)
    
    if args.test_only:
        step5_test_steering()
        return
    
    # Step 1: Train SAE
    if not args.skip_training:
        step1_train_sae(quick=args.quick)
    else:
        print("\nSkipping SAE training (using existing model)")
    
    # Step 2: Generate features
    features, item_ids = step2_generate_features()
    
    # Step 3: Analyze neurons
    step3_analyze_neurons(features, item_ids)
    
    # Step 4: Build text index
    step4_build_text_index()
    
    # Step 5: Test
    step5_test_steering()
    
    print("\n" + "="*60)
    print("PIPELINE REBUILD COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
