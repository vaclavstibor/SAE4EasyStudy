"""
Feature Steering Engine

Handles application of feature adjustments and recommendation reranking
based on user modifications to interpretable features.
"""

import numpy as np
from typing import List, Dict, Tuple


class FeatureAdjuster:
    """
    Applies user modifications to feature space.
    
    Supports different adjustment modes (additive, multiplicative, etc.)
    """
    
    def __init__(self, mode='multiplicative'):
        """
        Initialize adjuster.
        
        Args:
            mode: How to apply adjustments ('multiplicative', 'additive', 'absolute')
        """
        self.mode = mode
    
    def apply_adjustments(self, features, adjustments):
        """
        Apply feature adjustments.
        
        Args:
            features: Original feature activations (numpy array)
            adjustments: Dict mapping feature_id -> adjustment_value
        
        Returns:
            Modified features (numpy array)
        """
        modified = features.copy()
        
        for feature_id, adjustment in adjustments.items():
            feature_id = int(feature_id)
            
            if self.mode == 'multiplicative':
                # Multiply current activation by adjustment factor
                # adjustment=1.0 means no change, >1.0 increases, <1.0 decreases
                modified[feature_id] *= adjustment
                
            elif self.mode == 'additive':
                # Add adjustment to current activation
                modified[feature_id] += adjustment
                
            elif self.mode == 'absolute':
                # Set feature to absolute value
                modified[feature_id] = adjustment
        
        return modified
    
    def slider_to_adjustment(self, slider_value, mode='multiplicative'):
        """
        Convert slider position to adjustment value.
        
        Args:
            slider_value: Value from slider (e.g., -100 to 100, or 0 to 100)
            mode: Adjustment mode
        
        Returns:
            Adjustment value appropriate for the mode
        """
        if mode == 'multiplicative':
            # Map slider [-100, 100] to [0.0, 2.0]
            # -100 = 0.0 (turn off), 0 = 1.0 (no change), 100 = 2.0 (double)
            return (slider_value + 100) / 100.0
        
        elif mode == 'additive':
            # Slider value is directly added
            return slider_value / 100.0
        
        else:
            # For absolute mode, slider directly sets value
            return slider_value / 100.0
    
    def toggle_to_adjustment(self, is_enabled):
        """
        Convert toggle state to adjustment.
        
        Args:
            is_enabled: Boolean indicating if feature is enabled
        
        Returns:
            Adjustment value (0.0 if disabled, 1.0 if enabled)
        """
        return 1.0 if is_enabled else 0.0


class RecommendationReranker:
    """
    Reranks recommendations based on feature adjustments.
    
    Supports multiple reranking strategies.
    """
    
    def __init__(self, strategy='feature_weighted'):
        """
        Initialize reranker.
        
        Args:
            strategy: Reranking strategy ('feature_weighted', 'embedding_modification')
        """
        self.strategy = strategy
    
    def rerank_feature_weighted(
        self,
        base_scores: np.ndarray,
        item_features: np.ndarray,
        feature_adjustments: Dict[int, float],
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Rerank using feature-weighted score modification.
        
        Args:
            base_scores: Original recommendation scores (n_items,)
            item_features: Feature activations for items (n_items, n_features)
            feature_adjustments: Dict mapping feature_id -> adjustment_value
            alpha: Weight for feature adjustment term (0=ignore, 1=only features)
        
        Returns:
            Modified scores (n_items,)
        """
        # Start with base scores
        modified_scores = base_scores.copy()
        
        # Add weighted feature contributions
        feature_contribution = np.zeros(len(base_scores))
        
        for feature_id, adjustment in feature_adjustments.items():
            feature_id = int(feature_id)
            # How much each item activates this feature
            activations = item_features[:, feature_id]
            
            # Contribution is: (adjustment - 1.0) * activation
            # If adjustment=1.0 (no change), contribution=0
            # If adjustment=2.0 (double), add positive contribution
            # If adjustment=0.0 (turn off), add negative contribution
            contribution = (adjustment - 1.0) * activations
            feature_contribution += contribution
        
        # Combine base scores with feature contributions
        modified_scores = (1 - alpha) * base_scores + alpha * feature_contribution
        
        return modified_scores
    
    def rerank_embedding_modification(
        self,
        base_scores: np.ndarray,
        user_embedding: np.ndarray,
        item_embeddings: np.ndarray,
        modified_user_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Rerank by computing scores with modified user embedding.
        
        Args:
            base_scores: Original scores (for reference)
            user_embedding: Original user embedding
            item_embeddings: Item embeddings (n_items, embedding_dim)
            modified_user_embedding: User embedding after feature steering
        
        Returns:
            Modified scores (n_items,)
        """
        # Recompute scores as dot product with modified embedding
        modified_scores = np.dot(item_embeddings, modified_user_embedding)
        return modified_scores
    
    def rerank(
        self,
        items: List[int],
        base_scores: np.ndarray,
        item_features: np.ndarray,
        feature_adjustments: Dict[int, float],
        **kwargs
    ) -> List[Tuple[int, float]]:
        """
        Rerank items based on strategy.
        
        Args:
            items: List of item IDs
            base_scores: Base recommendation scores
            item_features: Feature activations for items
            feature_adjustments: User's feature adjustments
            **kwargs: Additional arguments for specific strategies
        
        Returns:
            List of (item_id, score) tuples, sorted by score descending
        """
        if self.strategy == 'feature_weighted':
            modified_scores = self.rerank_feature_weighted(
                base_scores,
                item_features,
                feature_adjustments,
                alpha=kwargs.get('alpha', 0.3)
            )
        
        elif self.strategy == 'embedding_modification':
            modified_scores = self.rerank_embedding_modification(
                base_scores,
                kwargs['user_embedding'],
                kwargs['item_embeddings'],
                kwargs['modified_user_embedding']
            )
        
        else:
            # Unknown strategy, return original
            modified_scores = base_scores
        
        # Sort by score descending
        sorted_indices = np.argsort(modified_scores)[::-1]
        
        result = [
            (items[idx], float(modified_scores[idx]))
            for idx in sorted_indices
        ]
        
        return result


class SteeringEngine:
    """
    Main steering engine coordinating feature adjustments and reranking.
    """
    
    def __init__(
        self,
        feature_extractor,
        adjustment_mode='multiplicative',
        reranking_strategy='feature_weighted'
    ):
        """
        Initialize steering engine.
        
        Args:
            feature_extractor: FeatureExtractor instance
            adjustment_mode: How to apply feature adjustments
            reranking_strategy: Strategy for reranking recommendations
        """
        self.feature_extractor = feature_extractor
        self.adjuster = FeatureAdjuster(mode=adjustment_mode)
        self.reranker = RecommendationReranker(strategy=reranking_strategy)
    
    def apply_steering(
        self,
        user_embedding: np.ndarray,
        candidate_items: List[int],
        item_embeddings: np.ndarray,
        item_features: np.ndarray,
        base_scores: np.ndarray,
        feature_adjustments: Dict[int, float]
    ) -> List[Tuple[int, float]]:
        """
        Apply steering and return reranked recommendations.
        
        Args:
            user_embedding: User's embedding vector
            candidate_items: List of candidate item IDs
            item_embeddings: Embeddings for candidate items
            item_features: Feature activations for candidate items
            base_scores: Base recommendation scores
            feature_adjustments: User's feature adjustments
        
        Returns:
            List of (item_id, score) tuples, reranked
        """
        # Option 1: Modify user embedding and recompute
        if self.reranker.strategy == 'embedding_modification':
            modified_user_embedding = self.feature_extractor.modify_embedding(
                user_embedding,
                feature_adjustments
            )
            
            reranked = self.reranker.rerank(
                candidate_items,
                base_scores,
                item_features,
                feature_adjustments,
                user_embedding=user_embedding,
                item_embeddings=item_embeddings,
                modified_user_embedding=modified_user_embedding
            )
        
        # Option 2: Rerank based on feature contributions
        else:
            reranked = self.reranker.rerank(
                candidate_items,
                base_scores,
                item_features,
                feature_adjustments
            )
        
        return reranked
    
    def get_feature_impact(
        self,
        feature_id: int,
        item_features: np.ndarray,
        base_scores: np.ndarray,
        adjustment_value: float
    ) -> Dict[str, float]:
        """
        Estimate impact of adjusting a feature.
        
        Args:
            feature_id: ID of feature to adjust
            item_features: Feature activations for items
            base_scores: Current scores
            adjustment_value: Proposed adjustment
        
        Returns:
            Dict with impact metrics
        """
        # Compute what scores would be with this adjustment
        test_adjustments = {feature_id: adjustment_value}
        modified_scores = self.reranker.rerank_feature_weighted(
            base_scores,
            item_features,
            test_adjustments
        )
        
        # Compute metrics
        score_change = np.mean(np.abs(modified_scores - base_scores))
        rank_change = self._compute_rank_change(base_scores, modified_scores)
        
        return {
            "avg_score_change": float(score_change),
            "avg_rank_change": float(rank_change),
            "max_score_change": float(np.max(np.abs(modified_scores - base_scores)))
        }
    
    def _compute_rank_change(self, original_scores, modified_scores):
        """Compute average rank change between two score lists"""
        original_ranks = np.argsort(np.argsort(original_scores)[::-1])
        modified_ranks = np.argsort(np.argsort(modified_scores)[::-1])
        return np.mean(np.abs(original_ranks - modified_ranks))
