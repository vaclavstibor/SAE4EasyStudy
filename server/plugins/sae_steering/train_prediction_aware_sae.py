"""
Prediction-Aware SAE Training for ELSA

Based on insights from:
- Arviv et al. (AAAI 2026) "Extracting Interaction-Aware Monosemantic Concepts"
- JumpReLU SAE (Rajamanoharan et al. 2024)
- BatchTopK SAE (Bussmann et al. 2024)

Key Innovation: SAE trained with prediction-aware loss that backpropagates
through frozen ELSA model, ensuring reconstructed embeddings preserve
recommendation behavior.

Architecture:
    ELSA item embedding (64-dim) → SAE (2048 neurons, k=32) → Reconstructed embedding
    
Loss = α * MSE(x, x_hat) + β * Prediction_Loss + γ * Sparsity_Penalty

Where Prediction_Loss ensures:
    ELSA.predict(user, item_original) ≈ ELSA.predict(user, item_reconstructed)

Usage:
    python train_prediction_aware_sae.py
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Configuration
INPUT_DIM = 64          # ELSA embedding dimension
SAE_HIDDEN_DIM = 2048   # SAE feature dictionary size
SAE_K = 32              # Top-k sparsity
BATCH_SIZE = 256
EPOCHS = 300            # Increased for better convergence
LR = 5e-4               # Reduced for stability

# Loss weights (tuned for interpretability + steering)
RECON_WEIGHT = 1.0           # L2 reconstruction weight
PREDICTION_WEIGHT = 0.05     # Prediction-aware loss (lower = more interpretable)
SPARSITY_WEIGHT = 0.01       # KL divergence sparsity weight
SELECTIVITY_WEIGHT = 0.5     # Selectivity loss (higher = more genre-specific neurons)

SPARSITY_TARGET = 0.02       # Target average activation per neuron (k/hidden = 32/2048)


class TopKActivation(nn.Module):
    """Top-K activation function with proper gradient handling."""
    
    def __init__(self, k: int):
        super().__init__()
        self.k = k
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = F.relu(topk.values)  # Only positive activations
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result


class TiedTranspose(nn.Module):
    """Tied weights decoder - uses transpose of encoder weights."""
    
    def __init__(self, encoder: nn.Linear):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.encoder.weight.t())
    
    @property
    def weight(self) -> torch.Tensor:
        return self.encoder.weight.t()


class PredictionAwareSAE(nn.Module):
    """
    Sparse Autoencoder with prediction-aware training.
    
    Key features:
    - TopK activation for exact sparsity control
    - Tied weights (decoder = encoder.T)
    - Pre-bias for input centering
    - Dead neuron tracking and resampling
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, k: int, tied: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.tied = tied
        
        # Pre-bias (learned input centering)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # Activation
        self.activation = TopKActivation(k)
        
        # Decoder (tied or separate)
        if tied:
            self.decoder = TiedTranspose(self.encoder)
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        
        # Track neuron usage for dead neuron detection
        self.register_buffer('neuron_usage', torch.zeros(hidden_dim))
        self.register_buffer('usage_count', torch.tensor(0.0))
        
        # Initialize
        self._init_weights()
        
        # Track training metrics
        self.loss_history = []
        
    def _init_weights(self):
        """Kaiming initialization."""
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        if hasattr(self.decoder, 'weight') and not isinstance(self.decoder, TiedTranspose):
            nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity='relu')
    
    def encode_pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to pre-activation latent space."""
        x_centered = x - self.pre_bias
        return F.linear(x_centered, self.encoder.weight, self.latent_bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to sparse latent space."""
        pre_act = self.encode_pre_activation(x)
        return self.activation(pre_act)
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(latents) + self.pre_bias
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Full forward pass.
        
        Returns:
            pre_act: Pre-activation latent values
            latents: Sparse latent activations
            reconstructed: Reconstructed input
        """
        pre_act = self.encode_pre_activation(x)
        latents = self.activation(pre_act)
        reconstructed = self.decode(latents)
        
        # Track neuron usage during training
        if self.training:
            with torch.no_grad():
                usage = (latents > 0).float().sum(dim=0)
                self.neuron_usage = self.neuron_usage * 0.99 + usage * 0.01
                self.usage_count += 1
        
        return pre_act, latents, reconstructed
    
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparse feature activations for inference."""
        with torch.no_grad():
            return self.encode(x)
    
    def get_dead_neurons(self, threshold: float = 1e-4) -> torch.Tensor:
        """Find neurons that rarely activate."""
        if self.usage_count < 100:
            return torch.tensor([])
        avg_usage = self.neuron_usage / (self.usage_count + 1e-8)
        dead_mask = avg_usage < threshold
        return torch.where(dead_mask)[0]
    
    def resample_dead_neurons(self, data_batch: torch.Tensor) -> int:
        """Reinitialize dead neurons using high-loss examples."""
        dead_neurons = self.get_dead_neurons()
        if len(dead_neurons) == 0:
            return 0
        
        with torch.no_grad():
            # Forward pass to find high-loss examples
            _, latents, reconstructed = self.forward(data_batch)
            losses = (reconstructed - data_batch).pow(2).sum(dim=1)
            
            # Get examples with highest reconstruction error
            num_to_resample = min(len(dead_neurons), len(data_batch))
            _, high_loss_indices = torch.topk(losses, num_to_resample)
            
            # Reinitialize dead neurons
            for i, neuron_idx in enumerate(dead_neurons[:num_to_resample]):
                example_idx = high_loss_indices[i % len(high_loss_indices)]
                direction = data_batch[example_idx] - self.pre_bias
                direction = direction / (direction.norm() + 1e-8)
                
                # Set encoder weight to point toward high-loss example
                self.encoder.weight.data[neuron_idx] = direction * 0.5
                self.latent_bias.data[neuron_idx] = 0.0
            
            # Reset usage tracking for resampled neurons
            self.neuron_usage[dead_neurons[:num_to_resample]] = 0.1
        
        return num_to_resample


def kl_divergence_loss(activations: torch.Tensor, target: float = 0.02) -> torch.Tensor:
    """
    KL divergence sparsity loss.
    
    Encourages average activation per neuron to be close to target.
    Returns MEAN (not sum) to keep scale reasonable.
    """
    # Average activation per neuron across batch (normalize by max to get probability-like values)
    act_normalized = activations / (activations.max() + 1e-8)
    rho_hat = act_normalized.mean(dim=0).clamp(1e-8, 1 - 1e-8)
    rho = torch.full_like(rho_hat, target)
    
    # KL(rho || rho_hat) - use mean not sum
    kl = rho * torch.log(rho / rho_hat) + \
         (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    
    return kl.mean()  # Mean instead of sum!


def selectivity_loss(activations: torch.Tensor, target_selectivity: float = 0.15) -> torch.Tensor:
    """
    Selectivity loss - penalizes neurons that activate on too many items.
    
    Args:
        activations: (batch, hidden_dim) tensor
        target_selectivity: Target fraction of items per neuron (0.15 = 15%)
    
    A selective neuron should activate on small fraction of items.
    For good genre separation, we want neurons to be active on ~10-20% of movies.
    """
    # Fraction of items each neuron activates on
    activation_rate = (activations > 0).float().mean(dim=0)  # (hidden_dim,)
    
    # Penalize neurons that exceed target selectivity (squared for stronger penalty)
    excess = F.relu(activation_rate - target_selectivity)
    
    return (excess ** 2).mean()  # Squared penalty for stronger effect


def load_elsa_model():
    """Load frozen ELSA model for prediction-aware training."""
    from train_elsa import ELSA, latent_dim
    
    with open(DATA_DIR / "item2index.pkl", "rb") as f:
        item2index = pickle.load(f)
    
    num_items = len(item2index)
    
    elsa = ELSA(num_items, latent_dim)
    elsa.load_state_dict(torch.load(MODELS_DIR / "elsa_model_best.pt", map_location='cpu'))
    elsa.eval()
    
    # Freeze ELSA
    for param in elsa.parameters():
        param.requires_grad = False
    
    return elsa, item2index


def compute_prediction_loss(elsa, item_embeddings_orig, item_embeddings_recon, 
                           user_indices, device):
    """
    Compute prediction-aware loss.
    
    Ensures reconstructed embeddings produce similar predictions to original.
    
    For ELSA: prediction = user_embedding @ item_embedding.T
    """
    # Sample random users for computing predictions
    num_users = elsa.A.shape[0]  # Actually items, but we use them as "user" preferences
    
    # Get a subset of "user" embeddings (other items as context)
    sample_size = min(100, num_users)
    user_sample = torch.randint(0, num_users, (sample_size,), device=device)
    
    # Get user embeddings from ELSA
    user_emb = F.normalize(elsa.A[user_sample], dim=1)  # (sample, 64)
    
    # Original predictions: user_emb @ item_orig.T
    pred_orig = torch.matmul(user_emb, item_embeddings_orig.T)  # (sample, batch)
    
    # Reconstructed predictions: user_emb @ item_recon.T  
    pred_recon = torch.matmul(user_emb, item_embeddings_recon.T)  # (sample, batch)
    
    # MSE between predictions
    prediction_loss = F.mse_loss(pred_recon, pred_orig)
    
    return prediction_loss


def train_prediction_aware_sae():
    """Main training function with prediction-aware loss."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Load ELSA and data
    print("Loading ELSA model...")
    elsa, item2index = load_elsa_model()
    elsa = elsa.to(device)
    
    # Get normalized item embeddings
    item_embeddings = F.normalize(elsa.A.detach(), dim=1)
    num_items = item_embeddings.shape[0]
    print(f"Item embeddings: {item_embeddings.shape}")
    
    # Create SAE
    sae = PredictionAwareSAE(
        input_dim=INPUT_DIM,
        hidden_dim=SAE_HIDDEN_DIM,
        k=SAE_K,
        tied=True
    ).to(device)
    
    print(f"\nSAE config:")
    print(f"  Input dim: {INPUT_DIM}")
    print(f"  Hidden dim: {SAE_HIDDEN_DIM}")
    print(f"  Top-k: {SAE_K}")
    print(f"  Tied weights: True")
    
    # Create dataset
    dataset = TensorDataset(
        item_embeddings,
        torch.arange(num_items)  # Item indices for prediction loss
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
    
    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    
    best_loss = float("inf")
    best_active = 0
    
    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"Loss weights: recon={RECON_WEIGHT}, pred={PREDICTION_WEIGHT}, sparse={SPARSITY_WEIGHT}, select={SELECTIVITY_WEIGHT}")
    
    for epoch in range(EPOCHS):
        sae.train()
        total_loss = 0
        total_recon = 0
        total_pred = 0
        total_sparse = 0
        total_select = 0
        
        for batch_emb, batch_idx in loader:
            batch_emb = batch_emb.to(device)
            batch_idx = batch_idx.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pre_act, latents, reconstructed = sae(batch_emb)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, batch_emb)
            
            # Prediction-aware loss (lower weight now)
            pred_loss = compute_prediction_loss(
                elsa, 
                batch_emb, 
                reconstructed,
                batch_idx,
                device
            )
            
            # Sparsity loss (KL divergence)
            sparse_loss = kl_divergence_loss(latents, SPARSITY_TARGET)
            
            # Selectivity loss (encourage neurons to be selective - activate on <15% of items)
            select_loss = selectivity_loss(latents, target_selectivity=0.15)
            
            # Total loss
            loss = (RECON_WEIGHT * recon_loss + 
                   PREDICTION_WEIGHT * pred_loss + 
                   SPARSITY_WEIGHT * sparse_loss +
                   SELECTIVITY_WEIGHT * select_loss)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_pred += pred_loss.item()
            total_sparse += sparse_loss.item()
            total_select += select_loss.item()
        
        scheduler.step()
        
        # Resample dead neurons every 10 epochs
        if (epoch + 1) % 10 == 0:
            sample_idx = torch.randperm(len(item_embeddings))[:BATCH_SIZE]
            sample_batch = item_embeddings[sample_idx].to(device)
            num_resampled = sae.resample_dead_neurons(sample_batch)
            if num_resampled > 0:
                print(f"  [Epoch {epoch+1}] Resampled {num_resampled} dead neurons")
        
        avg_loss = total_loss / len(loader)
        avg_recon = total_recon / len(loader)
        avg_pred = total_pred / len(loader)
        avg_sparse = total_sparse / len(loader)
        avg_select = total_select / len(loader)
        
        sae.loss_history.append(avg_loss)
        
        # Evaluation
        sae.eval()
        with torch.no_grad():
            sample_emb = item_embeddings[:1000].to(device)
            latents = sae.get_feature_activations(sample_emb)
            
            active_features = (latents > 0).any(dim=0).sum().item()
            avg_active = (latents > 0).sum(dim=1).float().mean().item()
        
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{EPOCHS}: "
                  f"Loss={avg_loss:.4f} (recon={avg_recon:.4f}, pred={avg_pred:.4f}, sel={avg_select:.3f}), "
                  f"Active={active_features}/{SAE_HIDDEN_DIM}, Avg/item={avg_active:.1f}, LR={lr:.2e}")
        
        # Save best model (based on reconstruction, not prediction - we want selectivity!)
        if avg_recon < best_loss and active_features > 100:  # Require reasonable diversity
            best_loss = avg_recon
            best_active = active_features
            
            MODELS_DIR.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': sae.state_dict(),
                'config': {
                    'input_dim': INPUT_DIM,
                    'hidden_dim': SAE_HIDDEN_DIM,
                    'k': SAE_K,
                    'tied': True,
                },
                'epoch': epoch,
                'loss': avg_loss,
                'active_features': active_features,
            }, MODELS_DIR / "prediction_aware_sae.pt")
    
    print(f"\n Training complete!")
    print(f"Best: loss={best_loss:.4f}, active_features={best_active}")
    print(f"Model saved to: {MODELS_DIR / 'prediction_aware_sae.pt'}")
    
    return sae


def analyze_prediction_aware_sae(model_path=None):
    """Analyze features learned by prediction-aware SAE."""
    
    if model_path is None:
        model_path = MODELS_DIR / "prediction_aware_sae.pt"
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    sae = PredictionAwareSAE(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        k=config['k'],
        tied=config.get('tied', True)
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Active features: {checkpoint['active_features']}")
    
    # Load ELSA and data
    elsa, item2index = load_elsa_model()
    item_embeddings = F.normalize(elsa.A.detach(), dim=1)
    
    # Load movie metadata - MovieLens as base, TMDB as enrichment
    import json
    import pandas as pd
    DATASET_DIR = BASE_DIR.parent.parent / "static" / "datasets" / "ml-latest"
    
    # Load MovieLens movies.csv (base - has all movies)
    movies = {}
    movies_path = DATASET_DIR / "movies.csv"
    if movies_path.exists():
        df = pd.read_csv(movies_path)
        for _, row in df.iterrows():
            movie_id = str(row['movieId'])
            movies[movie_id] = {
                'title': row['title'],
                'genres': row['genres'].split('|') if pd.notna(row['genres']) else []
            }
        print(f"Loaded {len(movies)} movies from MovieLens")
    
    # Enrich with TMDB data (has cast, keywords, overview)
    tmdb_file = DATASET_DIR / "tmdb_data.json"
    if tmdb_file.exists():
        with open(tmdb_file, 'r', encoding='utf-8') as f:
            tmdb_data = json.load(f)
        
        enriched = 0
        for movie_id, tmdb_info in tmdb_data.items():
            if movie_id in movies:
                # Add TMDB fields to existing entry
                movies[movie_id]['cast'] = tmdb_info.get('cast', [])
                movies[movie_id]['keywords'] = tmdb_info.get('keywords', [])
                movies[movie_id]['overview'] = tmdb_info.get('overview', '')
                movies[movie_id]['directors'] = tmdb_info.get('directors', [])
                enriched += 1
        print(f"Enriched {enriched} movies with TMDB data")
    
    # Get activations for all items
    with torch.no_grad():
        all_latents = sae.get_feature_activations(item_embeddings)
    
    # Statistics
    active_per_item = (all_latents > 0).sum(dim=1).float()
    active_features = (all_latents > 0).any(dim=0).sum().item()
    
    print(f"\nStatistics:")
    print(f"  Total features: {config['hidden_dim']}")
    print(f"  Active features: {active_features}")
    print(f"  Avg active per item: {active_per_item.mean():.1f}")
    
    # Find most important features
    feature_importance = all_latents.sum(dim=0)
    top_features = torch.topk(feature_importance, min(20, active_features)).indices
    
    index2item = {v: k for k, v in item2index.items()}
    
    print("\nTop 20 most active features:")
    print("=" * 80)
    
    for feat_idx in top_features:
        feat_idx = feat_idx.item()
        
        # Find items that activate this feature
        feature_activations = all_latents[:, feat_idx]
        num_active = (feature_activations > 0).sum().item()
        top_items = torch.topk(feature_activations, min(5, num_active)).indices
        
        print(f"\nFeature {feat_idx} (items: {num_active}, total_act: {feature_importance[feat_idx]:.1f}):")
        
        for item_idx in top_items:
            item_idx = item_idx.item()
            movie_id = index2item[item_idx]
            
            # Get movie info (MovieLens base + TMDB enrichment)
            movie_info = movies.get(str(movie_id), {})
            title = movie_info.get('title', f'Movie {movie_id}')
            genres = movie_info.get('genres', [])
            
            activation = feature_activations[item_idx].item()
            print(f"  {title} [{', '.join(genres[:3])}] - {activation:.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true", help="Analyze trained model")
    args = parser.parse_args()
    
    if args.analyze:
        analyze_prediction_aware_sae()
    else:
        train_prediction_aware_sae()
