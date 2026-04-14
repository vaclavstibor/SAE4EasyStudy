"""
SAE-based Recommendation Engine

Generates recommendations using SAE feature steering.
Takes neuron adjustments and returns items that match the desired features.

Supports:
- Basic SAE (sae_model_r4_k32.pt)           – EasyStudy native
- Prediction-aware SAE (prediction_aware_sae.pt) – EasyStudy native (RECOMMENDED)
- WWW TopKSAE / BasicSAE checkpoints        – from WWW_disentangling
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

# Import SAE architectures
try:
    from .train_sae import TopKSAE
    from .train_prediction_aware_sae import PredictionAwareSAE
    from .www_models import TopKSAE_WWW, BasicSAE as BasicSAE_WWW, load_www_checkpoint
    from .model_store import (
        DEFAULT_TOPK_SAE_MODEL_ID,
        find_local_model_path,
        format_missing_model_message,
    )
except ImportError:
    from train_sae import TopKSAE
    from train_prediction_aware_sae import PredictionAwareSAE
    from www_models import TopKSAE_WWW, BasicSAE as BasicSAE_WWW, load_www_checkpoint
    from model_store import (
        DEFAULT_TOPK_SAE_MODEL_ID,
        find_local_model_path,
        format_missing_model_message,
    )

DEFAULT_WWW_MODEL_ID = DEFAULT_TOPK_SAE_MODEL_ID


class SAERecommender:
    """
    Recommender that uses SAE feature space for steering.
    
    Flow:
    1. User provides feature adjustments (neuron_id -> weight)
    2. Create "ideal" feature vector from adjustments
    3. Find items whose SAE activations best match this ideal
    
    Supports multiple SAE models for A/B comparison.
    """
    
    def __init__(self, data_dir: str = None, model_id: str = None):
        """
        Initialize SAE Recommender.
        
        Args:
            data_dir: Path to directory containing SAE model and data files
            model_id: Specific SAE model to load (e.g., 'www_TopKSAE_8192', 'prediction_aware_sae')
                     If None, uses the default WWW TopK model when available.
        """
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "data"
        )
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        self.model_id = model_id  # Store for A/B comparison
        
        self.sae_model = None
        self.item_features = None  # Pre-computed SAE activations for all items
        self.item_ids = None       # Mapping from index to movie_id
        self.item_embeddings = None
        self.neuron_labels = None
        self.selective_neurons = None
        
        self._loaded = False

    @property
    def sae_features(self):
        """Alias so callers using ``recommender.sae_features`` keep working."""
        return self.item_features
    
    def load(self):
        """Load SAE model and pre-computed features."""
        if self._loaded:
            return
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        resolved_model_id = self.model_id or DEFAULT_WWW_MODEL_ID
        local_model_path = find_local_model_path(resolved_model_id)
        if local_model_path is None:
            raise FileNotFoundError(format_missing_model_message(resolved_model_id))

        sae_path = str(local_model_path)
        features_cache_name = f"item_sae_features_{resolved_model_id}.pt"
        
        state_dict = torch.load(sae_path, map_location=device, weights_only=False)

        # --- Detect checkpoint format ----------------------------------------
        # WWW_disentangling: {epoch, job_cfg, model_state_dict, optimizer_state_dict}
        # EasyStudy native:  {model_state_dict, config, ...} or raw state_dict
        is_www = isinstance(state_dict, dict) and 'job_cfg' in state_dict

        if is_www:
            self.sae_model, www_cfg = load_www_checkpoint(sae_path, device)
            input_dim = www_cfg.get("input_dim", www_cfg.get("embedding_dim"))
            hidden_dim = www_cfg.get("embedding_dim")
            print(f"Loading WWW {www_cfg['model_class']}: input={input_dim}, hidden={hidden_dim}")
        else:
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                model_state = state_dict['model_state_dict']
                config = state_dict.get('config', {})
            else:
                model_state = state_dict
                config = {}

            input_dim = model_state['encoder.weight'].shape[1]
            hidden_dim = model_state['encoder.weight'].shape[0]
            k = config.get('k', 32)

            is_prediction_aware = 'pre_bias' in model_state or 'latent_bias' in model_state

            if is_prediction_aware:
                self.sae_model = PredictionAwareSAE(input_dim, hidden_dim, k, tied=config.get('tied', True)).to(device)
                print(f"Loading Prediction-Aware SAE: input={input_dim}, hidden={hidden_dim}, k={k}")
            else:
                self.sae_model = TopKSAE(input_dim, hidden_dim, k).to(device)
                print(f"Loading Basic SAE: input={input_dim}, hidden={hidden_dim}, k={k}")

            self.sae_model.load_state_dict(model_state)
            self.sae_model.eval()
        
        # Load item embeddings (from ELSA)
        embeddings_path = os.path.join(self.data_dir, "item_embeddings.pt")
        if os.path.exists(embeddings_path):
            data = torch.load(embeddings_path, map_location=device, weights_only=False)
            self.item_embeddings = data['embeddings']
            self.item_ids = data['item_ids']
            print(f"Loaded {len(self.item_ids)} item embeddings")
        else:
            # Try to load from text_embeddings.pt which has item_ids
            text_emb_path = os.path.join(self.data_dir, "text_embeddings.pt")
            if os.path.exists(text_emb_path):
                data = torch.load(text_emb_path, map_location=device, weights_only=False)
                self.item_ids = data['item_ids']
                print(f"Loaded item_ids from text_embeddings.pt: {len(self.item_ids)} items")
                
                # Need to load ELSA embeddings separately
                elsa_path = os.path.join(self.data_dir, "elsa_embeddings.pt")
                if os.path.exists(elsa_path):
                    self.item_embeddings = torch.load(elsa_path, map_location=device, weights_only=False)
        
        # Pre-compute SAE activations if not cached
        features_path = os.path.join(self.data_dir, features_cache_name)
        if os.path.exists(features_path):
            data = torch.load(features_path, map_location=device, weights_only=False)
            self.item_features = data['features']
            # Always prefer item_ids from the feature cache (they match the features matrix)
            if 'item_ids' in data and data['item_ids'] is not None:
                self.item_ids = data['item_ids']
            if self.item_ids is not None and len(self.item_ids) != int(self.item_features.shape[0]):
                print(
                    "[SAERecommender.load] WARNING: item_ids length does not match feature rows "
                    f"({len(self.item_ids)} vs {int(self.item_features.shape[0])})"
                )
            print(f"Loaded pre-computed SAE features: {self.item_features.shape}"
                  f" ({len(self.item_ids)} item_ids)")
        elif self.item_embeddings is not None:
            print("Computing SAE features for all items...")
            self._compute_item_features(features_cache_name)
        
        # Load neuron labels
        labels_path = os.path.join(self.data_dir, "neuron_labels.json")
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.neuron_labels = json.load(f)
            print(f"Loaded {len(self.neuron_labels)} neuron labels")
        
        # Load selective neurons
        analysis_path = os.path.join(self.data_dir, "neuron_analysis.json")
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r') as f:
                analysis = json.load(f)
            # The keys are the selective neuron IDs
            self.selective_neurons = set(int(k) for k in analysis.keys() if k.isdigit())
            print(f"Loaded {len(self.selective_neurons)} selective neurons")
        
        self._loaded = True
        # Store cache filename for later use
        self._features_cache_name = features_cache_name
    
    def _compute_item_features(self, cache_filename="item_sae_features.pt"):
        """Compute SAE activations for all items."""
        if self.sae_model is None:
            raise RuntimeError("Cannot compute features: model not loaded")
        
        device = next(self.sae_model.parameters()).device
        
        # ELSA is a CF model: features are the encoder columns, not SAE activations
        from .www_models import ELSA_WWW
        if isinstance(self.sae_model, ELSA_WWW):
            with torch.no_grad():
                # For ELSA: item embeddings are the L2-normalised encoder columns
                # Shape: (input_dim=num_items, embedding_dim) — each row is one item
                raw_emb = self.sae_model.get_item_embeddings()  # (n_items, emb_dim)
                self.item_features = F.relu(raw_emb)  # non-negative "activations"
                # item_ids are simply 0..n_items-1 indices in ELSA's item space
                if self.item_ids is None or len(self.item_ids) != raw_emb.shape[0]:
                    self.item_ids = list(range(raw_emb.shape[0]))
            print(f"Computed ELSA item features: {self.item_features.shape}")
        else:
            if self.item_embeddings is None:
                raise RuntimeError("Cannot compute features: embeddings not loaded")
            # Normalize embeddings (same as training)
            embeddings_norm = F.normalize(self.item_embeddings.to(device), dim=1)
            # Get SAE activations
            with torch.no_grad():
                self.item_features = self.sae_model.get_feature_activations(embeddings_norm)
        
        # Cache for future use
        cache_path = os.path.join(self.data_dir, cache_filename)
        torch.save({
            'features': self.item_features.cpu(),
            'item_ids': self.item_ids
        }, cache_path)
        print(f"Cached SAE features to {cache_path}")
    
    def get_recommendations(
        self,
        feature_adjustments: Dict[int, float],
        n_items: int = 20,
        exclude_items: List[int] = None,
        method: str = 'weighted_match',
        allowed_ids: Optional[set] = None,
        seed_adjustments: Optional[Dict[int, float]] = None,
    ) -> List[Dict]:
        """
        Generate recommendations based on feature adjustments.

        When *seed_adjustments* is provided the scoring uses a two-component
        blend so that user slider changes gradually override the seed profile:

            score = (1 - beta) * seed_score  +  beta * user_score

        beta is derived from the total magnitude of the user's own slider
        moves (excluding the seed).  This means a small slider change only
        slightly perturbs the seed ranking, and a full-range move can
        completely replace it — giving visible incremental steering.

        Without a seed, a mean-activation baseline takes the role of the
        "seed" so single-slider moves are also meaningful (instead of
        producing identical rankings regardless of magnitude).
        """
        self.load()

        if self.item_features is None:
            return []

        exclude_set = set(exclude_items or [])
        allowed_set = set(allowed_ids) if allowed_ids is not None else set(self.item_ids)
        device = self.item_features.device

        def _to_weights(adj: Dict[int, float]) -> torch.Tensor:
            w = torch.zeros(self.item_features.shape[1], device=device)
            for nid, val in adj.items():
                nid = int(nid)
                if 0 <= nid < len(w):
                    w[nid] = val
            return w

        def _score(weights: torch.Tensor) -> torch.Tensor:
            if method == 'cosine_similarity':
                ideal = F.normalize(weights.unsqueeze(0), dim=1)
                items = F.normalize(self.item_features, dim=1)
                return torch.matmul(items, ideal.T).squeeze()
            return torch.matmul(self.item_features, weights)

        def _minmax(t: torch.Tensor) -> torch.Tensor:
            r = t.max() - t.min()
            return (t - t.min()) / r if r > 0 else torch.zeros_like(t)

        def _user_normalize(scores: torch.Tensor) -> torch.Tensor:
            """Normalize user slider scores with symmetric handling.

            Positive-only steering:
            - activated items get a meaningful floor boost in [0.5, 1.0]
            - inactive items stay at 0.0

            Negative-only steering:
            - items matching the suppressed feature are pushed toward 0.0
            - items with near-zero activation rise toward 1.0

            Mixed positive/negative steering:
            - zero acts as a neutral midpoint (0.5)
            - positive scores map to (0.5, 1.0]
            - negative scores map to [0.0, 0.5)
            """
            pos = scores > 0
            neg = scores < 0

            if not pos.any() and not neg.any():
                return torch.zeros_like(scores)

            # Positive-only: keep the stronger "activation floor" behavior
            # so sparse TopK features visibly affect the ranking.
            if pos.any() and not neg.any():
                result = torch.zeros_like(scores)
                vals = scores[pos]
                p_min, p_max = vals.min(), vals.max()
                p_range = p_max - p_min
                if p_range > 0:
                    result[pos] = 0.5 + 0.5 * (vals - p_min) / p_range
                else:
                    result[pos] = 1.0
                return result

            # Negative-only: invert the ordering so that low-activation items
            # are favored and strongly matching items are suppressed.
            if neg.any() and not pos.any():
                return _minmax(scores)

            # Mixed signs: treat 0 as neutral midpoint.
            result = torch.full_like(scores, 0.5)

            pos_vals = scores[pos]
            p_min, p_max = pos_vals.min(), pos_vals.max()
            p_range = p_max - p_min
            if p_range > 0:
                result[pos] = 0.5 + 0.5 * (pos_vals - p_min) / p_range
            else:
                result[pos] = 1.0

            neg_vals = scores[neg]
            n_min = neg_vals.min()
            n_range = 0 - n_min
            if n_range > 0:
                result[neg] = 0.5 * (neg_vals - n_min) / n_range
            else:
                result[neg] = 0.0

            return result

        def _compose_scores(base_scores: torch.Tensor, user_weights: torch.Tensor, strength: float) -> torch.Tensor:
            """Compose base ranking with positive boost / negative suppression.

            Positive sliders add attraction toward matching items.
            Negative sliders subtract a penalty from items matching the
            suppressed feature, which yields a smooth "move away from this"
            effect instead of a sudden jump to arbitrary anti-feature items.
            """
            base_norm = _minmax(base_scores)
            pos_weights = torch.clamp(user_weights, min=0)
            neg_weights = torch.clamp(-user_weights, min=0)

            has_pos = bool((pos_weights > 0).any().item())
            has_neg = bool((neg_weights > 0).any().item())

            if not has_pos and not has_neg:
                return base_norm

            if has_pos and not has_neg:
                pos_scores = _score(pos_weights)
                return (1 - strength) * base_norm + strength * _user_normalize(pos_scores)

            if has_neg and not has_pos:
                neg_scores = _score(neg_weights)
                return base_norm - strength * _user_normalize(neg_scores)

            pos_scores = _score(pos_weights)
            neg_scores = _score(neg_weights)
            return base_norm + strength * (_user_normalize(pos_scores) - _user_normalize(neg_scores))

        if seed_adjustments:
            seed_weights = _to_weights(seed_adjustments)
            user_adj = {
                int(k): float(v) - float(seed_adjustments.get(int(k), seed_adjustments.get(str(k), 0)))
                for k, v in feature_adjustments.items()
            }
            user_adj = {k: v for k, v in user_adj.items() if abs(v) > 0.001}
            user_weights = _to_weights(user_adj)

            has_pos_user = bool((user_weights > 0).any().item())
            has_neg_user = bool((user_weights < 0).any().item())
            user_total = user_weights.abs().sum().item()

            user_override_scale = 1.5

            if user_total > 0.001:
                seed_scores = _score(seed_weights)
                beta = min(user_total / user_override_scale, 1.0)
                scores = _compose_scores(seed_scores, user_weights, beta)
            else:
                scores = _minmax(_score(seed_weights))
        else:
            feature_weights = _to_weights(feature_adjustments)
            baseline_scores = self.item_features.mean(dim=1)
            total_adjustment = feature_weights.abs().sum().item()
            alpha = min(total_adjustment, 1.0)
            scores = _compose_scores(baseline_scores, feature_weights, alpha)

        # Apply allowed_id mask
        allowed_mask = None
        if allowed_set is not None:
            allowed_mask = torch.tensor(
                [item_id in allowed_set for item_id in self.item_ids],
                device=device,
                dtype=torch.bool,
            )
            allowed_count = int(allowed_mask.sum().item())
            if allowed_count < max(25, int(0.5 * len(self.item_ids))):
                print(
                    "[SAERecommender.get_recommendations] WARNING: restrictive allowed_id mask "
                    f"({allowed_count}/{len(self.item_ids)} items allowed)"
                )
            if not allowed_mask.any():
                return []
            scores_allowed = scores[allowed_mask]
        else:
            scores_allowed = scores

        # Normalize to [0, 1] for display
        score_min = scores_allowed.min()
        score_max = scores_allowed.max()
        score_range = score_max - score_min

        if score_range > 0:
            norm_allowed = (scores_allowed - score_min) / score_range
        else:
            norm_allowed = torch.zeros_like(scores_allowed)

        if allowed_mask is not None:
            scores_normalized = torch.full_like(scores, float("-inf"))
            scores_normalized[allowed_mask] = norm_allowed
        else:
            scores_normalized = norm_allowed
        
        # Sort by score descending (higher scores = better match)
        sorted_indices = torch.argsort(scores_normalized, descending=True)
        
        results = []
        for idx in sorted_indices:
            idx = idx.item()
            item_id = self.item_ids[idx]
            if allowed_set is not None and item_id not in allowed_set:
                continue
            
            if item_id in exclude_set:
                continue
            
            # Get which features this item activates
            item_feats = self.item_features[idx]
            matched_features = {}
            
            for neuron_id, weight in feature_adjustments.items():
                neuron_id = int(neuron_id)
                activation = item_feats[neuron_id].item()
                if activation > 0:
                    label = (self.neuron_labels or {}).get(str(neuron_id), f"Feature {neuron_id}")
                    matched_features[label] = {
                        'activation': round(activation, 3),
                        'weight': round(weight, 3)
                    }
            
            # Use normalized score (0-1 range) for display
            results.append({
                'movie_id': int(item_id),
                'score': round(scores_normalized[idx].item(), 4),
                'matched_features': matched_features
            })
            
            if len(results) >= n_items:
                break
        
        return results
    
    def get_item_features(self, item_id: int) -> Dict[str, float]:
        """
        Get SAE feature activations for a specific item.
        
        Args:
            item_id: Movie ID
        
        Returns:
            Dict mapping feature_label -> activation
        """
        self.load()
        
        if self.item_features is None or self.item_ids is None:
            return {}
        
        try:
            idx = self.item_ids.index(item_id)
        except ValueError:
            return {}
        
        features = self.item_features[idx]
        
        result = {}
        # Get top-k active features
        topk_values, topk_indices = torch.topk(features, min(10, len(features)))
        
        for val, idx in zip(topk_values, topk_indices):
            if val.item() > 0:
                label = self.neuron_labels.get(str(idx.item()), f"Feature {idx.item()}")
                result[label] = round(val.item(), 3)
        
        return result


# Global instances - supports multiple models for A/B comparison
_sae_recommenders: Dict[str, SAERecommender] = {}
_default_recommender = None


def get_sae_recommender(model_id: str = None) -> SAERecommender:
    """
    Get or create SAE recommender instance.
    
    Args:
        model_id: Specific model to load. If None, uses default.
                 Supports: 'www_TopKSAE_8192', 'prediction_aware_sae', 'sae_model_r4_k32', etc.
    
    Returns:
        SAERecommender instance for the specified model
    """
    global _sae_recommenders, _default_recommender
    
    if model_id is None:
        # Return default recommender
        if _default_recommender is None:
            _default_recommender = SAERecommender(model_id=DEFAULT_WWW_MODEL_ID)
        return _default_recommender
    
    # Check cache for specific model
    if model_id not in _sae_recommenders:
        _sae_recommenders[model_id] = SAERecommender(model_id=model_id)
    
    return _sae_recommenders[model_id]


def generate_sae_recommendations(
    feature_adjustments: Dict[int, float],
    n_items: int = 20,
    exclude_items: List[int] = None,
    model_id: str = None,
    allowed_ids: Optional[set] = None,
) -> List[Dict]:
    """
    Convenience function to generate SAE-based recommendations.
    
    Args:
        feature_adjustments: Dict mapping neuron_id -> weight
        n_items: Number of recommendations
        exclude_items: Items to exclude
        model_id: Specific SAE model to use (for A/B testing)
    
    Returns:
        List of recommendation dicts
    """
    recommender = get_sae_recommender(model_id=model_id)
    return recommender.get_recommendations(
        feature_adjustments=feature_adjustments,
        n_items=n_items,
        exclude_items=exclude_items,
        allowed_ids=allowed_ids,
    )


def get_available_models() -> List[Dict]:
    """
    Get list of available SAE models for configuration.
    
    Returns:
        List of model info dicts
    """
    available = []
    
    model_files = [
        (DEFAULT_WWW_MODEL_ID, "WWW TopKSAE-8192 (k=32)", "Release-managed default checkpoint"),
        ("prediction_aware_sae", "Prediction-Aware SAE", "Optional local checkpoint"),
        ("sae_model_r4_k32", "Basic TopK SAE", "Optional local checkpoint"),
        ("multimodal_sae", "Multimodal SAE", "Optional local checkpoint"),
    ]
    
    for model_id, name, description in model_files:
        path = find_local_model_path(model_id)
        if path is not None:
            available.append({
                "id": model_id,
                "name": name,
                "description": description,
                "path": str(path)
            })
    
    return available
