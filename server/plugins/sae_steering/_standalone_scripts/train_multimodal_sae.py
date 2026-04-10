"""
Multimodal SAE Training Pipeline

Combines collaborative filtering embeddings (ELSA) with content embeddings (Sentence-BERT)
to create an SAE that understands both user preferences AND movie content.

Architecture:
    ELSA embedding (64-dim) + Text embedding (384-dim) → Combined (128-dim) → SAE (2048 neurons)

This allows the SAE to learn features like:
- "Tom Cruise action movies" (content-aware)
- "Movies liked by users who like Nolan films" (preference-aware)
- "Cerebral sci-fi" (combination of both)

Usage:
    python train_multimodal_sae.py
"""

import os
import json
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
DATASET_DIR = BASE_DIR.parent.parent / "static" / "datasets" / "ml-latest"

# Configuration
ELSA_DIM = 64           # ELSA embedding dimension
TEXT_DIM = 384          # Sentence-BERT (all-MiniLM-L6-v2) dimension
COMBINED_DIM = 256      # Combined embedding dimension (increased for more capacity)
SAE_HIDDEN_DIM = 2048   # SAE feature dictionary size
SAE_K = 64              # Top-k sparsity (increased to prevent collapse)
BATCH_SIZE = 256
EPOCHS = 200            # More epochs for better convergence
LR = 3e-4               # Lower learning rate for stability
DEAD_NEURON_THRESHOLD = 1e-5  # Threshold for considering a neuron dead
RESAMPLE_INTERVAL = 50  # How often to resample dead neurons


class MovieTextEncoder:
    """
    Encode movie metadata into text embeddings using Sentence-BERT.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = 384  # all-MiniLM-L6-v2 output dimension
    
    def create_movie_text(self, movie_data: dict) -> str:
        """Create rich text description from movie metadata."""
        parts = []
        
        # Title
        title = movie_data.get('title', '')
        if title:
            parts.append(title)
        
        # Genres
        genres = movie_data.get('genres', [])
        if isinstance(genres, str):
            genres = genres.split('|')
        if genres:
            parts.append(f"Genres: {', '.join(genres)}")
        
        # Directors
        directors = movie_data.get('directors', [])
        if directors:
            parts.append(f"Directed by {', '.join(directors[:2])}")
        
        # Cast (top 3)
        cast = movie_data.get('cast', [])
        if cast:
            actor_names = [a['name'] if isinstance(a, dict) else a for a in cast[:3]]
            parts.append(f"Starring {', '.join(actor_names)}")
        
        # Keywords (top 5)
        keywords = movie_data.get('keywords', [])
        if keywords:
            parts.append(f"Keywords: {', '.join(keywords[:5])}")
        
        # Overview/plot (truncated)
        overview = movie_data.get('overview', '') or movie_data.get('plot', '')
        if overview:
            parts.append(overview[:200])
        
        return ' | '.join(parts) if parts else title
    
    def encode_movies(self, movie_data_dict: dict, item2index: dict) -> torch.Tensor:
        """
        Encode all movies into text embeddings.
        
        Args:
            movie_data_dict: Dict mapping tmdb_id or movie_id to movie metadata
            item2index: Dict mapping movieId to SAE item index
        
        Returns:
            Tensor of shape (num_items, text_dim)
        """
        num_items = len(item2index)
        embeddings = torch.zeros(num_items, self.dim)
        
        # Create reverse mapping
        index2item = {v: k for k, v in item2index.items()}
        
        # Try to find movie data by various keys
        texts = []
        valid_indices = []
        
        for idx in tqdm(range(num_items), desc="Preparing movie texts"):
            movie_id = index2item[idx]
            
            # Try different keys
            movie_data = None
            for key in [str(movie_id), movie_id]:
                if key in movie_data_dict:
                    movie_data = movie_data_dict[key]
                    break
            
            if movie_data:
                text = self.create_movie_text(movie_data)
                texts.append(text)
                valid_indices.append(idx)
            else:
                # Fallback: just use movie ID
                texts.append(f"Movie {movie_id}")
                valid_indices.append(idx)
        
        # Batch encode
        print(f"Encoding {len(texts)} movie descriptions...")
        all_embeddings = self.model.encode(
            texts, 
            batch_size=64, 
            show_progress_bar=True,
            convert_to_tensor=True
        )
        
        for i, idx in enumerate(valid_indices):
            embeddings[idx] = all_embeddings[i]
        
        return embeddings


class CombinedEncoder(nn.Module):
    """
    Combines ELSA and text embeddings into a unified representation.
    
    Uses learned projections to balance collaborative and content signals.
    Architecture improved with residual connections and better normalization.
    """
    
    def __init__(self, elsa_dim=ELSA_DIM, text_dim=TEXT_DIM, output_dim=COMBINED_DIM):
        super().__init__()
        
        self.elsa_dim = elsa_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        
        # Project both modalities to same dimension first
        hidden_dim = output_dim // 2
        
        # ELSA projection with residual-like structure
        self.elsa_proj = nn.Sequential(
            nn.Linear(elsa_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Text projection 
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, elsa_emb, text_emb):
        """
        Combine ELSA and text embeddings.
        
        Args:
            elsa_emb: (batch, elsa_dim)
            text_emb: (batch, text_dim)
        
        Returns:
            combined: (batch, output_dim)
        """
        elsa_proj = self.elsa_proj(elsa_emb)
        text_proj = self.text_proj(text_emb)
        
        # Concatenate
        combined = torch.cat([elsa_proj, text_proj], dim=-1)
        
        # Final fusion
        combined = self.fusion(combined)
        
        # L2 normalize
        combined = F.normalize(combined, dim=-1)
        
        return combined


class MultimodalTopKSAE(nn.Module):
    """
    Sparse Autoencoder that operates on combined embeddings.
    
    Includes the combined encoder as part of the model.
    Features improved initialization and dead neuron handling.
    """
    
    def __init__(self, elsa_dim, text_dim, combined_dim, hidden_dim, k):
        super().__init__()
        
        self.combined_encoder = CombinedEncoder(elsa_dim, text_dim, combined_dim)
        
        self.k = k
        self.hidden_dim = hidden_dim
        self.combined_dim = combined_dim
        
        # SAE encoder/decoder
        self.encoder = nn.Linear(combined_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, combined_dim, bias=True)
        
        # Better initialization: Kaiming for encoder, tied-ish for decoder
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
        
        # Initialize decoder as transpose of encoder (approximately tied weights)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
        nn.init.zeros_(self.decoder.bias)
        
        # Track neuron usage for dead neuron detection
        self.register_buffer('neuron_usage', torch.zeros(hidden_dim))
        self.register_buffer('usage_count', torch.tensor(0))
    
    def topk_activation(self, h):
        """Apply top-k sparsity."""
        topk_values, topk_indices = torch.topk(h, self.k, dim=-1)
        h_sparse = torch.zeros_like(h)
        h_sparse.scatter_(-1, topk_indices, F.relu(topk_values))
        return h_sparse, topk_indices
    
    def forward(self, elsa_emb, text_emb):
        """
        Full forward pass.
        
        Returns:
            reconstructed: Reconstructed combined embedding
            h_sparse: Sparse hidden activations
            combined: Combined input embedding
            topk_indices: Which neurons were activated
        """
        # Combine modalities
        combined = self.combined_encoder(elsa_emb, text_emb)
        
        # SAE forward
        h_pre = self.encoder(combined)
        h_sparse, topk_indices = self.topk_activation(h_pre)
        reconstructed = self.decoder(h_sparse)
        
        # Track neuron usage
        if self.training:
            with torch.no_grad():
                usage = (h_sparse > 0).float().sum(dim=0)
                self.neuron_usage = self.neuron_usage * 0.99 + usage * 0.01
                self.usage_count += 1
        
        return reconstructed, h_sparse, combined, topk_indices
    
    def get_feature_activations(self, elsa_emb, text_emb):
        """Get sparse feature activations."""
        with torch.no_grad():
            combined = self.combined_encoder(elsa_emb, text_emb)
            h_pre = self.encoder(combined)
            h_sparse, _ = self.topk_activation(h_pre)
        return h_sparse
    
    def get_dead_neurons(self, threshold=DEAD_NEURON_THRESHOLD):
        """Find neurons that rarely activate."""
        if self.usage_count < 100:
            return torch.tensor([])
        avg_usage = self.neuron_usage / (self.usage_count + 1e-8)
        dead_mask = avg_usage < threshold
        return torch.where(dead_mask)[0]
    
    def resample_dead_neurons(self, data_batch):
        """Reinitialize dead neurons using high-loss examples."""
        dead_neurons = self.get_dead_neurons()
        if len(dead_neurons) == 0:
            return 0
        
        elsa_emb, text_emb = data_batch
        with torch.no_grad():
            combined = self.combined_encoder(elsa_emb, text_emb)
            h_pre = self.encoder(combined)
            h_sparse, _ = self.topk_activation(h_pre)
            reconstructed = self.decoder(h_sparse)
            
            # Find examples with highest reconstruction error
            losses = (reconstructed - combined).pow(2).sum(dim=1)
            _, high_loss_indices = torch.topk(losses, min(len(dead_neurons), len(losses)))
            
            # Reinitialize dead neurons to point toward high-loss examples
            for i, neuron_idx in enumerate(dead_neurons[:len(high_loss_indices)]):
                example_idx = high_loss_indices[i % len(high_loss_indices)]
                direction = combined[example_idx]
                
                # Set encoder to detect this direction
                self.encoder.weight.data[neuron_idx] = direction * 0.5
                self.encoder.bias.data[neuron_idx] = 0.0
                
                # Set decoder to reconstruct this direction
                self.decoder.weight.data[:, neuron_idx] = direction
            
            # Reset usage tracking for resampled neurons
            self.neuron_usage[dead_neurons] = 0.1
        
        return len(dead_neurons)


def load_all_data():
    """Load ELSA embeddings, item mappings, and movie metadata."""
    
    print("Loading data...")
    
    # Load ELSA model and extract embeddings
    from train_elsa import ELSA, latent_dim
    
    with open(DATA_DIR / "item2index.pkl", "rb") as f:
        item2index = pickle.load(f)
    
    num_items = len(item2index)
    
    elsa = ELSA(num_items, latent_dim)
    elsa.load_state_dict(torch.load(MODELS_DIR / "elsa_model_best.pt", map_location='cpu'))
    elsa.eval()
    
    elsa_embeddings = F.normalize(elsa.A.detach(), dim=1)
    print(f"  ELSA embeddings: {elsa_embeddings.shape}")
    
    # Load movie metadata (from enrichment)
    tmdb_file = DATASET_DIR / "tmdb_data.json"
    if tmdb_file.exists():
        with open(tmdb_file, 'r', encoding='utf-8') as f:
            tmdb_data = json.load(f)
        print(f"  TMDB metadata: {len(tmdb_data)} movies")
    else:
        print("  ⚠ No TMDB metadata found, using empty dict")
        tmdb_data = {}
    
    return elsa_embeddings, item2index, tmdb_data


def train_multimodal_sae():
    """Main training function with improved dead neuron handling."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Load data
    elsa_embeddings, item2index, tmdb_data = load_all_data()
    num_items = len(item2index)
    
    # Create text embeddings
    print("\nCreating text embeddings...")
    text_encoder = MovieTextEncoder()
    text_embeddings = text_encoder.encode_movies(tmdb_data, item2index)
    text_embeddings = F.normalize(text_embeddings, dim=1)
    print(f"  Text embeddings: {text_embeddings.shape}")
    
    # Create model
    model = MultimodalTopKSAE(
        elsa_dim=ELSA_DIM,
        text_dim=TEXT_DIM,
        combined_dim=COMBINED_DIM,
        hidden_dim=SAE_HIDDEN_DIM,
        k=SAE_K
    ).to(device)
    
    print(f"\nModel config:")
    print(f"  ELSA dim: {ELSA_DIM}")
    print(f"  Text dim: {TEXT_DIM}")
    print(f"  Combined dim: {COMBINED_DIM}")
    print(f"  SAE hidden: {SAE_HIDDEN_DIM}")
    print(f"  Top-k: {SAE_K}")
    
    # Create dataset
    dataset = TensorDataset(elsa_embeddings, text_embeddings)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Optimizer with weight decay only on non-bias parameters
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': 1e-4},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=LR)
    
    # Learning rate schedule: warmup then cosine decay
    def lr_lambda(epoch):
        warmup_epochs = 10
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_loss = float("inf")
    best_active_features = 0
    
    print(f"\nTraining for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_recon = 0
        total_aux = 0
        
        for batch_idx, (elsa_batch, text_batch) in enumerate(loader):
            elsa_batch = elsa_batch.to(device)
            text_batch = text_batch.to(device)
            
            optimizer.zero_grad()
            
            reconstructed, h_sparse, combined, topk_indices = model(elsa_batch, text_batch)
            
            # Reconstruction loss (main objective)
            recon_loss = F.mse_loss(reconstructed, combined)
            
            # Auxiliary loss: encourage diversity in neuron usage
            # Compute batch-level neuron usage
            batch_usage = (h_sparse > 0).float().mean(dim=0)  # (hidden_dim,)
            
            # Penalize if some neurons are used too much (prevents collapse)
            usage_var = batch_usage.var()
            diversity_loss = -usage_var * 0.1  # Maximize variance in usage
            
            # Small L1 on activations to encourage true sparsity
            l1_loss = h_sparse.abs().mean() * 0.0001
            
            loss = recon_loss + diversity_loss + l1_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_aux += (diversity_loss + l1_loss).item()
        
        scheduler.step()
        
        # Resample dead neurons periodically
        if (epoch + 1) % RESAMPLE_INTERVAL == 0:
            # Get a random batch for resampling
            sample_idx = torch.randperm(len(elsa_embeddings))[:BATCH_SIZE]
            sample_batch = (
                elsa_embeddings[sample_idx].to(device),
                text_embeddings[sample_idx].to(device)
            )
            num_resampled = model.resample_dead_neurons(sample_batch)
            if num_resampled > 0:
                print(f"  Resampled {num_resampled} dead neurons")
        
        avg_loss = total_loss / len(loader)
        avg_recon = total_recon / len(loader)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Use larger sample for better estimate
            sample_size = min(1000, len(elsa_embeddings))
            sample_idx = torch.randperm(len(elsa_embeddings))[:sample_size]
            sample_elsa = elsa_embeddings[sample_idx].to(device)
            sample_text = text_embeddings[sample_idx].to(device)
            
            h_sparse = model.get_feature_activations(sample_elsa, sample_text)
            
            active_features = (h_sparse > 0).any(dim=0).sum().item()
            avg_active = (h_sparse > 0).sum(dim=1).float().mean().item()
            
            # Dead neurons count
            dead_neurons = model.get_dead_neurons()
            num_dead = len(dead_neurons)
        
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{EPOCHS}: Loss={avg_loss:.6f} (recon={avg_recon:.6f}), "
                  f"Active={active_features}/{SAE_HIDDEN_DIM}, Dead={num_dead}, "
                  f"Avg/item={avg_active:.1f}, LR={lr:.2e}")
        
        # Save best model (prefer more active features)
        score = -avg_recon + active_features * 0.0001  # Balance loss and diversity
        if active_features > best_active_features or (active_features == best_active_features and avg_recon < best_loss):
            best_loss = avg_recon
            best_active_features = active_features
            MODELS_DIR.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'elsa_dim': ELSA_DIM,
                    'text_dim': TEXT_DIM,
                    'combined_dim': COMBINED_DIM,
                    'hidden_dim': SAE_HIDDEN_DIM,
                    'k': SAE_K,
                },
                'epoch': epoch,
                'active_features': active_features,
            }, MODELS_DIR / "multimodal_sae_v2.pt")
    
    print(f"\nBest: loss={best_loss:.6f}, active_features={best_active_features}")
    print(f"Model saved to: {MODELS_DIR / 'multimodal_sae_v2.pt'}")
    
    # Save text embeddings for inference
    torch.save({
        'embeddings': text_embeddings,
        'item_ids': list(item2index.keys())
    }, DATA_DIR / "text_embeddings_v2.pt")
    print(f"Text embeddings saved to: {DATA_DIR / 'text_embeddings_v2.pt'}")
    
    return model


def analyze_multimodal_features(model_path=None):
    """Analyze what features the multimodal SAE learned."""
    
    if model_path is None:
        model_path = MODELS_DIR / "multimodal_sae_v2.pt"
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get config from checkpoint or use defaults
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        state_dict = checkpoint['model_state_dict']
    else:
        config = {
            'elsa_dim': ELSA_DIM,
            'text_dim': TEXT_DIM,
            'combined_dim': COMBINED_DIM,
            'hidden_dim': SAE_HIDDEN_DIM,
            'k': SAE_K,
        }
        state_dict = checkpoint
    
    # Load model
    model = MultimodalTopKSAE(
        elsa_dim=config['elsa_dim'],
        text_dim=config['text_dim'],
        combined_dim=config['combined_dim'],
        hidden_dim=config['hidden_dim'],
        k=config['k']
    )
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Loaded model with config: {config}")
    
    # Load data
    elsa_embeddings, item2index, tmdb_data = load_all_data()
    
    # Try to load v2 text embeddings first
    text_emb_path = DATA_DIR / "text_embeddings_v2.pt"
    if text_emb_path.exists():
        data = torch.load(text_emb_path, map_location='cpu')
        text_embeddings = data['embeddings'] if isinstance(data, dict) else data
    else:
        text_embeddings = torch.load(DATA_DIR / "text_embeddings.pt", map_location='cpu')
        if isinstance(text_embeddings, dict):
            text_embeddings = text_embeddings['embeddings']
    
    # Get activations for all items
    with torch.no_grad():
        h_sparse = model.get_feature_activations(elsa_embeddings, text_embeddings)
    
    # Statistics
    active_per_item = (h_sparse > 0).sum(dim=1).float()
    active_features = (h_sparse > 0).any(dim=0).sum().item()
    
    print(f"\nStatistics:")
    print(f"  Total features: {config['hidden_dim']}")
    print(f"  Active features (used at least once): {active_features}")
    print(f"  Average active per item: {active_per_item.mean():.1f}")
    
    # Find top features
    feature_importance = h_sparse.sum(dim=0)
    top_features = torch.topk(feature_importance, min(20, active_features)).indices
    
    index2item = {v: k for k, v in item2index.items()}
    
    print("\nTop 20 most active features:")
    print("=" * 80)
    
    for feat_idx in top_features:
        feat_idx = feat_idx.item()
        
        # Find items that activate this feature
        feature_activations = h_sparse[:, feat_idx]
        num_active = (feature_activations > 0).sum().item()
        top_items = torch.topk(feature_activations, min(5, num_active)).indices
        
        print(f"\nFeature {feat_idx} (total act: {feature_importance[feat_idx]:.2f}, "
              f"items: {num_active}):")
        
        for item_idx in top_items:
            item_idx = item_idx.item()
            movie_id = index2item[item_idx]
            
            # Get movie info
            movie_info = tmdb_data.get(str(movie_id), {})
            title = movie_info.get('title', f'Movie {movie_id}')
            genres = movie_info.get('genres', [])
            if isinstance(genres, str):
                genres = genres.split('|')
            
            # Get cast
            cast = movie_info.get('cast', [])
            cast_names = [c['name'] if isinstance(c, dict) else c for c in cast[:2]]
            
            activation = feature_activations[item_idx].item()
            print(f"  {title} [{', '.join(genres[:2])}] "
                  f"({', '.join(cast_names)}) - {activation:.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true", help="Analyze trained model")
    args = parser.parse_args()
    
    if args.analyze:
        analyze_multimodal_features()
    else:
        train_multimodal_sae()
