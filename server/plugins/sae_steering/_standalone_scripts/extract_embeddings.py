"""
Extract and save ELSA embeddings for SAE Recommender

This script extracts item embeddings from the ELSA model and saves them
in a format usable by the SAE recommender.
"""

import pickle
import torch
import torch.nn.functional as F
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def extract_embeddings():
    """Extract ELSA item embeddings and save them."""
    print("Loading item2index mapping...")
    with open(DATA_DIR / "item2index.pkl", "rb") as f:
        item2index = pickle.load(f)
    
    print(f"Found {len(item2index)} items")
    
    # Import ELSA model
    from train_elsa import ELSA, latent_dim
    
    print(f"Loading ELSA model (latent_dim={latent_dim})...")
    elsa = ELSA(len(item2index), latent_dim)
    elsa.load_state_dict(torch.load(MODELS_DIR / "elsa_model_best.pt", map_location='cpu'))
    elsa.eval()
    
    # Extract embeddings (normalized)
    print("Extracting normalized embeddings...")
    embeddings = F.normalize(elsa.A.detach(), dim=1)
    
    # Create item_ids list (order must match embedding indices)
    item_ids = [None] * len(item2index)
    for item_id, idx in item2index.items():
        item_ids[idx] = int(item_id)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Item IDs: {len(item_ids)} items, first 5: {item_ids[:5]}")
    
    # Save
    output_path = DATA_DIR / "item_embeddings.pt"
    torch.save({
        'embeddings': embeddings,
        'item_ids': item_ids,
        'latent_dim': latent_dim
    }, output_path)
    
    print(f"Saved embeddings to {output_path}")
    
    # Also compute and save SAE features
    print("\nComputing SAE features...")
    from train_sae import TopKSAE, hidden_dim, k
    
    sae = TopKSAE(latent_dim, hidden_dim, k)
    state_dict = torch.load(MODELS_DIR / "sae_model_r4_k32.pt", map_location='cpu')
    # Handle both formats: direct state_dict or checkpoint with 'model_state_dict' key
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        sae.load_state_dict(state_dict['model_state_dict'])
    else:
        sae.load_state_dict(state_dict)
    sae.eval()
    
    with torch.no_grad():
        features = sae.get_feature_activations(embeddings)
    
    print(f"SAE features shape: {features.shape}")
    
    # Save SAE features
    features_path = DATA_DIR / "item_sae_features.pt"
    torch.save({
        'features': features,
        'item_ids': item_ids
    }, features_path)
    
    print(f"Saved SAE features to {features_path}")
    
    # Print some stats
    print(f"\nStats:")
    print(f"  - Non-zero activations per item: {(features > 0).sum(dim=1).float().mean():.1f}")
    print(f"  - Active neurons: {(features.sum(dim=0) > 0).sum().item()}")
    
    return embeddings, features, item_ids


if __name__ == "__main__":
    extract_embeddings()
