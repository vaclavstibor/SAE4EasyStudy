"""
TopK Sparse Autoencoder

SAE with top-k activation function for exact sparsity control.
Trained on ELSA item embeddings to extract interpretable features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

# Configuration
k = 32              # Number of active features per item
hidden_dim = 2048   # Feature dictionary size (overcomplete)
batch_size = 512
epochs = 100
lr = 1e-3


class TopKSAE(nn.Module):
    """
    Sparse Autoencoder with Top-K activation.
    
    Architecture:
        Input (latent_dim) -> Encoder -> TopK -> Decoder -> Output (latent_dim)
    
    The top-k activation ensures exactly k features are active.
    """
    
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Decoder (tied weights optional - here we use separate)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
        # Initialize
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x):
        """Encode to pre-activation hidden representation."""
        return self.encoder(x)
    
    def topk_activation(self, h):
        """Apply top-k sparsity: keep only k largest activations."""
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(h, self.k, dim=-1)
        
        # Create sparse activation
        h_sparse = torch.zeros_like(h)
        h_sparse.scatter_(-1, topk_indices, F.relu(topk_values))
        
        return h_sparse
    
    def decode(self, h_sparse):
        """Decode from sparse representation."""
        return self.decoder(h_sparse)
    
    def forward(self, x):
        """
        Full forward pass.
        
        Returns:
            reconstructed: Reconstructed input
            h_sparse: Sparse hidden activations
            h_pre: Pre-activation hidden values (for auxiliary losses)
        """
        h_pre = self.encode(x)
        h_sparse = self.topk_activation(h_pre)
        reconstructed = self.decode(h_sparse)
        
        return reconstructed, h_sparse, h_pre
    
    def get_feature_activations(self, x):
        """Get sparse feature activations for input."""
        with torch.no_grad():
            h_pre = self.encode(x)
            h_sparse = self.topk_activation(h_pre)
        return h_sparse


def train_sae(embeddings, input_dim, save_path="models/sae_model_r4_k32.pt"):
    """
    Train TopK SAE on item embeddings.
    
    Args:
        embeddings: Item embeddings from ELSA (num_items, latent_dim)
        input_dim: Embedding dimension
        save_path: Where to save the model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Config: input_dim={input_dim}, hidden_dim={hidden_dim}, k={k}")
    
    # Normalize embeddings
    embeddings_norm = F.normalize(embeddings, dim=1)
    
    dataset = TensorDataset(embeddings_norm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = TopKSAE(input_dim, hidden_dim, k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_aux = 0
        
        for batch in loader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            
            reconstructed, h_sparse, h_pre = model(x)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, x)
            
            # Auxiliary loss: encourage pre-activations to be close to sparse activations
            # This helps with dead neurons
            aux_loss = F.mse_loss(h_pre, h_sparse.detach()) * 0.1
            
            loss = recon_loss + aux_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_aux += aux_loss.item()
        
        avg_loss = total_loss / len(loader)
        avg_recon = total_recon / len(loader)
        avg_aux = total_aux / len(loader)
        
        # Check sparsity
        model.eval()
        with torch.no_grad():
            sample = embeddings_norm[:100].to(device)
            _, h_sparse, _ = model(sample)
            active_ratio = (h_sparse > 0).float().mean().item()
            avg_active = (h_sparse > 0).sum(dim=1).float().mean().item()
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f} "
                  f"(recon={avg_recon:.6f}, aux={avg_aux:.6f}) "
                  f"Active={avg_active:.1f}/{hidden_dim}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
    
    print(f"\nBest loss: {best_loss:.6f}")
    return model


def analyze_features(model, embeddings, item2index, movies_df, top_features=10, top_items=5):
    """Analyze what each SAE feature represents."""
    
    device = next(model.parameters()).device
    index2item = {v: k for k, v in item2index.items()}
    movieid_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))
    
    embeddings_norm = F.normalize(embeddings, dim=1).to(device)
    
    with torch.no_grad():
        _, h_sparse, _ = model(embeddings_norm)
    
    h_sparse = h_sparse.cpu()
    
    # Find most important features (highest total activation)
    feature_importance = h_sparse.sum(dim=0)
    top_feature_indices = torch.topk(feature_importance, top_features).indices
    
    print(f"\nTop {top_features} most active features:")
    print("=" * 60)
    
    for feat_idx in top_feature_indices:
        feat_idx = feat_idx.item()
        
        # Find items that activate this feature most
        feature_activations = h_sparse[:, feat_idx]
        top_item_indices = torch.topk(feature_activations, top_items).indices
        
        print(f"\nFeature {feat_idx} (total activation: {feature_importance[feat_idx]:.2f}):")
        
        for item_idx in top_item_indices:
            item_idx = item_idx.item()
            activation = feature_activations[item_idx].item()
            
            if item_idx in index2item:
                movie_id = index2item[item_idx]
                title = movieid_to_title.get(movie_id, f"Movie {movie_id}")
                print(f"  {activation:.3f} - {title}")


if __name__ == "__main__":
    import os
    import pickle
    import pandas as pd
    from train_elsa import ELSA, latent_dim
    
    os.makedirs("models", exist_ok=True)
    
    # Load ELSA model and get embeddings
    print("Loading ELSA model...")
    
    with open("data/item2index.pkl", "rb") as f:
        item2index = pickle.load(f)
    
    num_items = len(item2index)
    
    elsa = ELSA(num_items, latent_dim)
    elsa.load_state_dict(torch.load("models/elsa_model_best.pt"))
    elsa.eval()
    
    print("Extracting item embeddings...")
    embeddings = elsa.get_item_embeddings()
    print(f"Embeddings shape: {embeddings.shape}")
    
    print(f"\nTraining TopK SAE (k={k}, hidden_dim={hidden_dim})...")
    model = train_sae(embeddings, latent_dim)
    
    # Analyze features
    print("\nAnalyzing learned features...")
    movies_df = pd.read_csv("/Users/vaclav.stibor/Library/CloudStorage/OneDrive-HomeCreditInternationala.s/Documents/EasyStudy/server/static/datasets/ml-latest/ml-latest/movies.csv")
    analyze_features(model, embeddings, item2index, movies_df)
    
    print("\nDone. Model saved to models/sae_model_r4_k32.pt")
