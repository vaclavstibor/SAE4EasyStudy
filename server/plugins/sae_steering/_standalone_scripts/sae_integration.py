"""
SAE Integration Layer

Handles loading SAE models, extracting features from embeddings,
and mapping between neural representations and interpretable feature space.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn


class SAEModel:
    """
    Wrapper for Sparse Autoencoder model.
    
    Supports encoding embeddings to sparse feature space and decoding back.
    """
    
    def __init__(self, input_dim, feature_dim, sparsity_coefficient=1e-3):
        """
        Initialize SAE model.
        
        Args:
            input_dim: Dimension of input embeddings
            feature_dim: Dimension of feature space (typically overcomplete)
            sparsity_coefficient: L1 penalty weight for sparsity
        """
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.sparsity_coefficient = sparsity_coefficient
        
        # Simple linear encoder/decoder for POC
        self.encoder = nn.Linear(input_dim, feature_dim)
        self.decoder = nn.Linear(feature_dim, input_dim)
        self.activation = nn.ReLU()
        
    def encode(self, embeddings):
        """
        Encode embeddings to sparse feature space.
        
        Args:
            embeddings: Tensor of shape (batch_size, input_dim)
            
        Returns:
            features: Tensor of shape (batch_size, feature_dim)
        """
        with torch.no_grad():
            features = self.activation(self.encoder(embeddings))
        return features
    
    def decode(self, features):
        """
        Decode features back to embedding space.
        
        Args:
            features: Tensor of shape (batch_size, feature_dim)
            
        Returns:
            reconstructed: Tensor of shape (batch_size, input_dim)
        """
        with torch.no_grad():
            reconstructed = self.decoder(features)
        return reconstructed
    
    def forward(self, embeddings):
        """
        Full forward pass: encode then decode.
        
        Args:
            embeddings: Tensor of shape (batch_size, input_dim)
            
        Returns:
            reconstructed: Tensor of shape (batch_size, input_dim)
            features: Tensor of shape (batch_size, feature_dim)
        """
        features = self.encode(embeddings)
        reconstructed = self.decode(features)
        return reconstructed, features
    
    def save(self, path):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'input_dim': self.input_dim,
            'feature_dim': self.feature_dim,
            'sparsity_coefficient': self.sparsity_coefficient
        }, path)
    
    def load(self, path):
        """Load model from disk"""
        checkpoint = torch.load(path)
        self.input_dim = checkpoint['input_dim']
        self.feature_dim = checkpoint['feature_dim']
        self.sparsity_coefficient = checkpoint['sparsity_coefficient']
        
        self.encoder = nn.Linear(self.input_dim, self.feature_dim)
        self.decoder = nn.Linear(self.feature_dim, self.input_dim)
        
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])


class FeatureExtractor:
    """
    Extracts interpretable features from user/item embeddings using SAE.
    """
    
    def __init__(self, sae_model, feature_labels=None):
        """
        Initialize feature extractor.
        
        Args:
            sae_model: SAEModel instance
            feature_labels: Optional dict mapping feature_id -> {"label": str, "description": str}
        """
        self.sae_model = sae_model
        self.feature_labels = feature_labels or {}
    
    def extract_user_features(self, user_embedding, top_k=10):
        """
        Extract top-k most active features for a user.
        
        Args:
            user_embedding: User embedding vector (numpy array or tensor)
            top_k: Number of top features to return
            
        Returns:
            List of dicts with feature info:
            [{"id": int, "label": str, "activation": float, "description": str}, ...]
        """
        if isinstance(user_embedding, np.ndarray):
            user_embedding = torch.from_numpy(user_embedding).float()
        
        if len(user_embedding.shape) == 1:
            user_embedding = user_embedding.unsqueeze(0)
        
        # Get feature activations
        features = self.sae_model.encode(user_embedding)
        features = features.squeeze().numpy()
        
        # Get top-k indices
        top_indices = np.argsort(features)[-top_k:][::-1]
        
        result = []
        for idx in top_indices:
            feature_info = {
                "id": int(idx),
                "activation": float(features[idx]),
                "label": self.feature_labels.get(idx, {}).get("label", f"Feature {idx}"),
                "description": self.feature_labels.get(idx, {}).get("description", ""),
            }
            result.append(feature_info)
        
        return result
    
    def extract_item_features(self, item_embedding, top_k=5):
        """
        Extract top-k most active features for an item.
        
        Similar to extract_user_features but for items.
        """
        return self.extract_user_features(item_embedding, top_k)
    
    def get_feature_examples(self, feature_id, item_embeddings, item_ids, top_k=5):
        """
        Find items that most strongly activate a specific feature.
        
        Args:
            feature_id: ID of the feature to analyze
            item_embeddings: Array of item embeddings (n_items, embedding_dim)
            item_ids: List of item IDs corresponding to embeddings
            top_k: Number of example items to return
            
        Returns:
            List of (item_id, activation) tuples
        """
        if isinstance(item_embeddings, np.ndarray):
            item_embeddings = torch.from_numpy(item_embeddings).float()
        
        # Get feature activations for all items
        features = self.sae_model.encode(item_embeddings)
        feature_activations = features[:, feature_id].numpy()
        
        # Get top-k items
        top_indices = np.argsort(feature_activations)[-top_k:][::-1]
        
        examples = [
            (item_ids[idx], float(feature_activations[idx]))
            for idx in top_indices
        ]
        
        return examples
    
    def modify_embedding(self, embedding, feature_adjustments):
        """
        Modify an embedding based on feature adjustments.
        
        Args:
            embedding: Original embedding (numpy array or tensor)
            feature_adjustments: Dict mapping feature_id -> adjustment_value
                                Adjustment is multiplicative (1.0 = no change)
        
        Returns:
            Modified embedding (same type as input)
        """
        is_numpy = isinstance(embedding, np.ndarray)
        
        if is_numpy:
            embedding = torch.from_numpy(embedding).float()
        
        if len(embedding.shape) == 1:
            embedding = embedding.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Encode to feature space
        features = self.sae_model.encode(embedding)
        
        # Apply adjustments
        for feature_id, adjustment in feature_adjustments.items():
            features[0, feature_id] *= adjustment
        
        # Decode back to embedding space
        modified_embedding = self.sae_model.decode(features)
        
        if squeeze_output:
            modified_embedding = modified_embedding.squeeze(0)
        
        if is_numpy:
            modified_embedding = modified_embedding.numpy()
        
        return modified_embedding
    
    def save_feature_labels(self, path):
        """Save feature labels to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.feature_labels, f)
    
    def load_feature_labels(self, path):
        """Load feature labels from disk"""
        with open(path, 'rb') as f:
            self.feature_labels = pickle.load(f)


def generate_feature_labels(sae_model, item_embeddings, item_metadata, method='auto'):
    """
    Generate human-readable labels for SAE features.
    
    Args:
        sae_model: Trained SAE model
        item_embeddings: Item embeddings (n_items, embedding_dim)
        item_metadata: DataFrame with item metadata (titles, genres, etc.)
        method: Labeling method ('auto', 'genre-based', 'manual')
    
    Returns:
        Dict mapping feature_id -> {"label": str, "description": str}
    """
    feature_labels = {}
    
    if method == 'auto':
        # Placeholder for automatic labeling
        # In real implementation, could use:
        # - Clustering items by feature activation
        # - Extracting common genres/tags
        # - Using language models to generate descriptions
        
        feature_extractor = FeatureExtractor(sae_model)
        
        for feature_id in range(sae_model.feature_dim):
            # Find items that activate this feature
            examples = feature_extractor.get_feature_examples(
                feature_id, 
                item_embeddings,
                item_metadata.index.tolist(),
                top_k=10
            )
            
            # Simple heuristic: use most common genre among top items
            # (This is very simplified for POC)
            feature_labels[feature_id] = {
                "label": f"Feature {feature_id}",
                "description": f"Activated by {len(examples)} items"
            }
    
    return feature_labels


def train_simple_sae(embeddings, feature_dim, epochs=100, lr=1e-3, sparsity_coef=1e-3):
    """
    Train a simple SAE model on embeddings.
    
    POC implementation for demonstration purposes.
    
    Args:
        embeddings: Training embeddings (n_samples, embedding_dim)
        feature_dim: Dimension of feature space
        epochs: Number of training epochs
        lr: Learning rate
        sparsity_coef: L1 regularization coefficient
    
    Returns:
        Trained SAEModel
    """
    input_dim = embeddings.shape[1]
    model = SAEModel(input_dim, feature_dim, sparsity_coef)
    
    embeddings_tensor = torch.from_numpy(embeddings).float()
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=lr
    )
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, features = model.forward(embeddings_tensor)
        
        # Reconstruction loss
        recon_loss = nn.MSELoss()(reconstructed, embeddings_tensor)
        
        # Sparsity loss (L1 on features)
        sparsity_loss = torch.mean(torch.abs(features)) * sparsity_coef
        
        # Total loss
        loss = recon_loss + sparsity_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
                  f"Recon: {recon_loss.item():.4f}, Sparsity: {sparsity_loss.item():.4f}")
    
    return model
