"""
ELSA - Embarrassingly Lightweight Sparse Autoencoder for Recommendations

Learns item embeddings from user-item interactions using a simple 
linear autoencoder with L2 normalization.
"""

import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from tqdm import tqdm

# Configuration
latent_dim = 64
batch_size = 256
epochs = 50
lr = 1e-3
weight_decay = 1e-5


class SparseMatrixDataset(Dataset):
    """
    Dataset that works with sparse matrices efficiently.
    Converts rows to dense only when accessed.
    """
    def __init__(self, sparse_matrix):
        """
        Args:
            sparse_matrix: scipy.sparse.csr_matrix of shape (num_users, num_items)
        """
        self.sparse_matrix = sparse_matrix
        self.num_users = sparse_matrix.shape[0]
    
    def __len__(self):
        return self.num_users
    
    def __getitem__(self, idx):
        # Convert single row from sparse to dense
        row = self.sparse_matrix[idx].toarray().flatten().astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(row)
        if norm > 0:
            row = row / norm
        return torch.from_numpy(row)


class ELSA(nn.Module):
    """
    ELSA: Linear autoencoder for collaborative filtering.
    
    Architecture:
        Input (num_items) -> A (num_items x latent_dim) -> A^T -> Output (num_items)
    
    The matrix A serves as item embeddings.
    """
    
    def __init__(self, num_items, latent_dim):
        super().__init__()
        self.num_items = num_items
        self.latent_dim = latent_dim
        
        # Item embedding matrix
        self.A = nn.Parameter(torch.randn(num_items, latent_dim) * 0.01)
    
    def forward(self, x):
        """
        Forward pass: project to latent space and reconstruct.
        
        Args:
            x: User interaction vector (batch_size, num_items)
        
        Returns:
            Reconstructed interaction scores
        """
        # Normalize embeddings
        A_norm = F.normalize(self.A, dim=1)
        
        # Encode: x @ A -> (batch_size, latent_dim)
        latent = x @ A_norm
        
        # Decode: latent @ A^T -> (batch_size, num_items)
        output = latent @ A_norm.t()
        
        return output
    
    def get_user_embedding(self, x):
        """Get user embedding from interaction vector."""
        A_norm = F.normalize(self.A, dim=1)
        return x @ A_norm
    
    def get_item_embeddings(self):
        """Get normalized item embeddings."""
        return F.normalize(self.A, dim=1).detach()


def load_movielens_data(data_dir=None):
    """Load MovieLens data and create interaction matrix."""
    
    # Set default path relative to project root
    if data_dir is None:
        # Get project root (3 levels up from this script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, "../../..")
        data_dir = os.path.join(project_root, "server/static/datasets/ml-latest/ml-latest")
    
    ratings_df = pd.read_csv(f"{data_dir}/ratings.csv")
    movies_df = pd.read_csv(f"{data_dir}/movies.csv")
    
    # Create user and item indices
    user_ids = ratings_df["userId"].unique()
    item_ids = ratings_df["movieId"].unique()
    
    user2index = {uid: idx for idx, uid in enumerate(user_ids)}
    item2index = {iid: idx for idx, iid in enumerate(item_ids)}
    
    # Save mappings to local data directory (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_output_dir = os.path.join(script_dir, "data")
    os.makedirs(data_output_dir, exist_ok=True)
    
    with open(os.path.join(data_output_dir, "user2index.pkl"), "wb") as f:
        pickle.dump(user2index, f)
    with open(os.path.join(data_output_dir, "item2index.pkl"), "wb") as f:
        pickle.dump(item2index, f)
    
    # Also copy movies.csv to data directory for convenience
    movies_output_path = os.path.join(data_output_dir, "movies.csv")
    if not os.path.exists(movies_output_path):
        shutil.copy(f"{data_dir}/movies.csv", movies_output_path)
    
    # Create sparse interaction matrix
    rows = [user2index[u] for u in ratings_df["userId"]]
    cols = [item2index[i] for i in ratings_df["movieId"]]
    
    # Binary interactions (or use ratings as weights)
    values = np.ones(len(rows))
    
    num_users = len(user_ids)
    num_items = len(item_ids)
    
    interaction_matrix = csr_matrix(
        (values, (rows, cols)), 
        shape=(num_users, num_items)
    )
    
    print(f"Users: {num_users}, Items: {num_items}")
    print(f"Interactions: {len(rows)}")
    print(f"Sparsity: {1 - len(rows) / (num_users * num_items):.4f}")
    
    return interaction_matrix, user2index, item2index


def train_elsa(interaction_matrix, num_items, save_path="models/elsa_model_best.pt"):
    """Train ELSA model using sparse matrix to save memory."""
    
    # Device selection: MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print(f"Training on: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Training on: {device}")
    else:
        device = torch.device("cpu")
        print(f"Training on: {device}")
    
    # Use custom dataset that works with sparse matrix
    dataset = SparseMatrixDataset(interaction_matrix)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues with sparse matrices
    )
    
    model = ELSA(num_items, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = batch.to(device)
            
            optimizer.zero_grad()
            
            output = model(x)
            
            # Reconstruction loss (MSE)
            loss = F.mse_loss(output, x)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model")
    
    return model


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    print("Loading data...")
    interaction_matrix, user2index, item2index = load_movielens_data()
    
    num_items = interaction_matrix.shape[1]
    
    print(f"\nTraining ELSA (latent_dim={latent_dim})...")
    model = train_elsa(interaction_matrix, num_items)
    
    print("\nDone. Model saved to models/elsa_model_best.pt")
