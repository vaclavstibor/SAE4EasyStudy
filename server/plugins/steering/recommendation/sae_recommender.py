"""
SAE-based Recommendation Engine

Generates recommendations using TopKSAE feature steering.
Takes neuron adjustments and returns items that match the desired features.
"""

import csv
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .blending import build_blend_plan
    from .model_store import (
        DEFAULT_TOPK_SAE_MODEL_ID,
        find_local_model_path,
        format_missing_model_message,
    )
except ImportError:
    from blending import build_blend_plan
    from model_store import (
        DEFAULT_TOPK_SAE_MODEL_ID,
        find_local_model_path,
        format_missing_model_message,
    )


class TopKSAE(nn.Module):
    """Top-K Sparse Autoencoder over CF embeddings."""

    def __init__(self, input_dim: int, embedding_dim: int, k: int):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.k = k
        self.encoder_w = nn.Parameter(torch.empty(input_dim, embedding_dim))
        self.encoder_b = nn.Parameter(torch.zeros(embedding_dim))
        self.decoder_w = nn.Parameter(torch.empty(embedding_dim, input_dim))
        self.decoder_b = nn.Parameter(torch.zeros(input_dim))
        nn.init.kaiming_uniform_(self.encoder_w)
        nn.init.kaiming_uniform_(self.decoder_w)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - x_mean
        x_std = x_centered.std(dim=-1, keepdim=True) + 1e-7
        x_norm = x_centered / x_std
        e_pre = F.relu((x_norm - self.decoder_b) @ self.encoder_w + self.encoder_b)
        topk = torch.topk(e_pre, self.k, dim=-1)
        return torch.zeros_like(e_pre).scatter(-1, topk.indices, topk.values)

    @torch.no_grad()
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse activations for each input row (used by SAERecommender)."""
        return self.encode(x)


def load_checkpoint(filepath: str, device: torch.device = None):
    """Load a TopKSAE checkpoint and return (model, config).

    Handles the train directory checkpoint format where ``input_dim``
    may be absent from ``job_cfg`` — it is inferred from the state dict.
    """
    device = device or torch.device("cpu")
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    cfg = ckpt["job_cfg"]

    if "input_dim" not in cfg:
        state = ckpt["model_state_dict"]
        cfg["input_dim"] = state["encoder_w"].shape[0]

    model = TopKSAE(
        input_dim=cfg["input_dim"],
        embedding_dim=cfg["embedding_dim"],
        k=cfg.get("k"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


class SAERecommender:
    """
    Recommender that uses SAE feature space for steering.

    Flow:
    1. User provides feature adjustments (neuron_id -> weight)
    2. Create "ideal" feature vector from adjustments
    3. Find items whose SAE activations best match this ideal
    """

    def __init__(self, data_dir: str = None, model_id: str = None):
        plugin_dir = os.path.dirname(os.path.dirname(__file__))
        self.server_dir = os.path.abspath(os.path.join(plugin_dir, "..", ".."))
        self.data_dir = data_dir or os.path.join(plugin_dir, "data")
        self.models_dir = os.path.join(plugin_dir, "models")
        self.model_id = model_id

        self.sae_model = None
        self.item_features = None
        self.item_ids = None
        self.item_embeddings = None
        self.neuron_labels = None
        self.selective_neurons = None

        self._loaded = False

    def _infer_training_item_ids(
        self,
        dataset_variant: str = "ml-32m-filtered",
    ) -> Optional[List[int]]:
        """Reconstruct item ordering exactly as in train directory.

        Training uses ``np.unique`` over **string** IDs from ratings.csv,
        which is lexicographic (not integer sort). If we map encoder rows
        to numerically sorted IDs, recommendations become semantically wrong.
        """
        ratings_path = os.path.normpath(
            os.path.join(
                self.server_dir,
                "static",
                "datasets",
                dataset_variant,
                "ratings.csv",
            )
        )
        if not os.path.exists(ratings_path):
            return None

        item_values: List[str] = []
        with open(ratings_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None
            item_col = (
                "movieId"
                if "movieId" in reader.fieldnames
                else "item_id"
                if "item_id" in reader.fieldnames
                else None
            )
            if item_col is None:
                return None
            for row in reader:
                item_values.append(str(row[item_col]))

        ordered = np.unique(np.array(item_values, dtype=str))
        item_ids: List[int] = []
        for value in ordered.tolist():
            try:
                item_ids.append(int(value))
            except Exception:
                continue
        return item_ids

    @property
    def sae_features(self):
        """Alias so callers using ``recommender.sae_features`` keep working."""
        return self.item_features

    def load(self):
        """Load SAE model and pre-computed features."""
        if self._loaded:
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        resolved_model_id = self.model_id or DEFAULT_TOPK_SAE_MODEL_ID
        local_model_path = find_local_model_path(resolved_model_id)
        if local_model_path is None:
            raise FileNotFoundError(format_missing_model_message(resolved_model_id))

        sae_path = str(local_model_path)
        features_cache_name = f"item_sae_features_{resolved_model_id}.pt"

        self.sae_model, sae_cfg = load_checkpoint(sae_path, device)
        input_dim = sae_cfg["input_dim"]
        hidden_dim = sae_cfg["embedding_dim"]
        print(
            f"Loaded {sae_cfg['model_class']}: input={input_dim}, "
            f"hidden={hidden_dim}, k={sae_cfg.get('k')}"
        )

        # Load item embeddings (from ELSA)
        embeddings_path = os.path.join(self.data_dir, "item_embeddings.pt")
        if os.path.exists(embeddings_path):
            data = torch.load(embeddings_path, map_location=device, weights_only=False)
            self.item_embeddings = data["embeddings"]
            self.item_ids = data["item_ids"]
            print(f"Loaded {len(self.item_ids)} item embeddings")

        # Pre-compute SAE activations if not cached
        features_path = os.path.join(self.data_dir, features_cache_name)
        if os.path.exists(features_path):
            data = torch.load(features_path, map_location=device, weights_only=False)
            self.item_features = data["features"]
            if "item_ids" in data and data["item_ids"] is not None:
                self.item_ids = data["item_ids"]
            if self.item_ids is not None and len(self.item_ids) != int(self.item_features.shape[0]):
                print(
                    "[SAERecommender.load] WARNING: item_ids length does not match feature rows "
                    f"({len(self.item_ids)} vs {int(self.item_features.shape[0])})"
                )
            print(
                f"Loaded pre-computed SAE features: {self.item_features.shape}"
                f" ({len(self.item_ids)} item_ids)"
            )
        elif self.item_embeddings is not None:
            print("Computing SAE features for all items...")
            self._compute_item_features(features_cache_name)

        # Canonicalize item_ids to training order (lexicographic np.unique over strings).
        canonical_item_ids = self._infer_training_item_ids()
        if canonical_item_ids is not None:
            n_rows = None
            if self.item_features is not None:
                n_rows = int(self.item_features.shape[0])
            elif self.item_embeddings is not None:
                n_rows = int(self.item_embeddings.shape[0])

            if n_rows is not None and len(canonical_item_ids) == n_rows:
                if self.item_ids is None or [int(x) for x in self.item_ids] != canonical_item_ids:
                    print(
                        "[SAERecommender.load] Re-aligning item_ids to training order "
                        "(lexicographic item_id ordering from ratings.csv)."
                    )
                self.item_ids = canonical_item_ids
            else:
                print(
                    "[SAERecommender.load] WARNING: canonical item_ids length mismatch "
                    f"({len(canonical_item_ids)} vs rows={n_rows}); keeping existing mapping."
                )

        self._loaded = True
        self._features_cache_name = features_cache_name

    def _compute_item_features(self, cache_filename="item_sae_features.pt"):
        """Compute SAE activations for all items."""
        if self.sae_model is None:
            raise RuntimeError("Cannot compute features: model not loaded")
        if self.item_embeddings is None:
            raise RuntimeError("Cannot compute features: item_embeddings.pt not loaded")

        device = next(self.sae_model.parameters()).device
        embeddings_norm = F.normalize(self.item_embeddings.to(device), dim=1)
        with torch.no_grad():
            self.item_features = self.sae_model.get_feature_activations(embeddings_norm)

        cache_path = os.path.join(self.data_dir, cache_filename)
        torch.save({"features": self.item_features.cpu(), "item_ids": self.item_ids}, cache_path)
        print(f"Cached SAE features to {cache_path}")

    def _build_rank_deltas(
        self,
        base_scores: torch.Tensor,
        final_scores: torch.Tensor,
        valid_mask: torch.Tensor,
        cf_scores: torch.Tensor,
        genre_scores: torch.Tensor,
        steering_scores: torch.Tensor,
        top_k: int = 5,
    ) -> Dict[str, List[Dict]]:
        """Compute top upward/downward rank movers for debug interpretability."""
        device = base_scores.device
        n = base_scores.shape[0]

        invalid = ~valid_mask
        base_masked = base_scores.clone()
        final_masked = final_scores.clone()
        base_masked[invalid] = float("-inf")
        final_masked[invalid] = float("-inf")

        base_sorted = torch.argsort(base_masked, descending=True)
        final_sorted = torch.argsort(final_masked, descending=True)

        large = int(1e9)
        base_rank = torch.full((n,), large, dtype=torch.long, device=device)
        final_rank = torch.full((n,), large, dtype=torch.long, device=device)
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return {"top_up": [], "top_down": []}

        base_rank[base_sorted[:valid_count]] = torch.arange(1, valid_count + 1, device=device)
        final_rank[final_sorted[:valid_count]] = torch.arange(1, valid_count + 1, device=device)
        rank_delta = base_rank - final_rank  # positive = moved up

        valid_indices = torch.where(valid_mask)[0]
        vd = rank_delta[valid_indices]
        up_order = torch.argsort(vd, descending=True)
        down_order = torch.argsort(vd, descending=False)

        def _pack(idx_tensor: torch.Tensor) -> Dict:
            idx = int(idx_tensor.item())
            return {
                "movie_id": int(self.item_ids[idx]),
                "rank_delta": int(rank_delta[idx].item()),
                "base_rank": int(base_rank[idx].item()),
                "final_rank": int(final_rank[idx].item()),
                "base_score": round(float(base_scores[idx].item()), 4),
                "final_score": round(float(final_scores[idx].item()), 4),
                "cf_score": round(float(cf_scores[idx].item()), 4),
                "genre_score": round(float(genre_scores[idx].item()), 4),
                "steering_score": round(float(steering_scores[idx].item()), 4),
            }

        top_up = []
        for pos in up_order.tolist():
            idx = valid_indices[pos]
            if int(rank_delta[idx].item()) <= 0:
                break
            top_up.append(_pack(idx))
            if len(top_up) >= top_k:
                break

        top_down = []
        for pos in down_order.tolist():
            idx = valid_indices[pos]
            if int(rank_delta[idx].item()) >= 0:
                break
            top_down.append(_pack(idx))
            if len(top_down) >= top_k:
                break

        return {"top_up": top_up, "top_down": top_down}

    def get_recommendations(
        self,
        feature_adjustments: Dict[int, float],
        n_items: int = 20,
        exclude_items: List[int] = None,
        allowed_ids: Optional[set] = None,
        seed_embedding: Optional[np.ndarray] = None,
        genre_bonus: Optional[np.ndarray] = None,
        **_kwargs,
    ) -> Union[List[Dict], Dict]:
        """Rank items using hybrid scoring.

        Three signals are blended:
        1. **ELSA collaborative filtering** — cosine similarity between
           each item's dense ELSA embedding and the *seed_embedding*
           (mean of elicitation/liked movies).  This is the main
           "find similar movies" signal.
        2. **Genre overlap** — a pre-computed per-item bonus from the
           caller (Jaccard similarity to the seed movies' genre set).
        3. **SAE steering** — ``item_sae_features @ adjustments``.
           Only non-zero when the user moves sliders or likes movies.
        """
        self.load()

        if self.item_features is None:
            return []

        exclude_set = set(exclude_items or [])
        allowed_set = set(allowed_ids) if allowed_ids is not None else set(self.item_ids)
        device = self.item_features.device
        n_items_total = self.item_features.shape[0]
        n_features = self.item_features.shape[1]
        blend_plan = build_blend_plan(feature_adjustments)

        # --- 1. ELSA collaborative-filtering score (cosine similarity) ---
        cf_scores = torch.zeros(n_items_total, device=device)
        if seed_embedding is not None and self.item_embeddings is not None:
            seed_t = torch.tensor(
                seed_embedding,
                device=device,
                dtype=self.item_embeddings.dtype,
            )
            item_norms = F.normalize(self.item_embeddings.to(device), dim=1)
            seed_norm = F.normalize(seed_t.unsqueeze(0), dim=1).squeeze(0)
            cf_scores = torch.matmul(item_norms, seed_norm) * blend_plan.cf_weight

        # --- 2. Genre overlap bonus ---
        genre_scores = torch.zeros(n_items_total, device=device)
        if genre_bonus is not None:
            genre_scores = (
                torch.tensor(
                    genre_bonus,
                    device=device,
                    dtype=torch.float32,
                )
                * blend_plan.genre_weight
            )

        # --- 3. SAE steering score (from sliders / like boosts) ---
        sae_profile = torch.zeros(n_features, device=device)
        has_adjustments = False
        for nid, val in feature_adjustments.items():
            nid = int(nid)
            if 0 <= nid < n_features and abs(float(val)) > 1e-6:
                sae_profile[nid] = float(val)
                has_adjustments = True

        base_scores = cf_scores + genre_scores
        steering_scores = torch.zeros(n_items_total, device=device)
        prior_tiebreak_scores = torch.zeros(n_items_total, device=device)
        adaptive_gamma = 0.0
        clamp_value = 0.0
        if has_adjustments:
            sae_scores = torch.matmul(self.item_features, sae_profile)

            allowed_mask_tmp = torch.tensor(
                [mid in allowed_set for mid in self.item_ids],
                device=device,
                dtype=torch.bool,
            )
            candidate_sae = sae_scores[allowed_mask_tmp]

            if blend_plan.strategy == "steering_primary":
                steering_scores = sae_scores
                if blend_plan.prior_tiebreak_weight > 0 and int(allowed_mask_tmp.sum().item()) > 0:
                    candidate_base = base_scores[allowed_mask_tmp]
                    base_min = torch.min(candidate_base)
                    base_max = torch.max(candidate_base)
                    base_span = float((base_max - base_min).item())
                    if base_span > 1e-6:
                        normalized_base = (base_scores - base_min) / (base_max - base_min)
                        prior_tiebreak_scores = normalized_base * blend_plan.prior_tiebreak_weight
                clamp_value = (
                    float(torch.max(torch.abs(candidate_sae)).item())
                    if int(candidate_sae.numel())
                    else 0.0
                )
                adaptive_gamma = 1.0
            elif int(candidate_sae.numel()) >= 10:
                candidate_base = base_scores[allowed_mask_tmp]
                q75_b = torch.quantile(candidate_base, 0.75).item()
                q25_b = torch.quantile(candidate_base, 0.25).item()
                q75_s = torch.quantile(candidate_sae, 0.75).item()
                q25_s = torch.quantile(candidate_sae, 0.25).item()
                iqr_s = max(q75_s - q25_s, 1e-6)
                p95_b = torch.quantile(candidate_base, 0.95).item()
                p05_b = torch.quantile(candidate_base, 0.05).item()
                p95_s = torch.quantile(candidate_sae, 0.95).item()
                p05_s = torch.quantile(candidate_sae, 0.05).item()
                max_abs_s = torch.max(torch.abs(candidate_sae)).item()
                base_iqr = q75_b - q25_b
                base_span = p95_b - p05_b
                sae_span = max(p95_s - p05_s, iqr_s, max_abs_s, 1e-6)
                if base_span < 1e-5:
                    adaptive_gamma = 1.0
                    clamp_value = sae_span
                else:
                    iqr_b = max(base_iqr, 1e-6)
                    raw_gamma = 0.30 * (iqr_b / iqr_s)
                    adaptive_gamma = float(np.clip(raw_gamma, 0.03, 0.35))
                    clamp_value = max(0.35 * base_span, 0.05 * sae_span)

                steering_scores = torch.clamp(
                    adaptive_gamma * sae_scores,
                    -clamp_value,
                    clamp_value,
                )
            else:
                adaptive_gamma = 0.15
                clamp_value = 2.0
                steering_scores = torch.clamp(
                    adaptive_gamma * sae_scores,
                    -clamp_value,
                    clamp_value,
                )

        scores = base_scores + steering_scores + prior_tiebreak_scores

        # Mask disallowed items with -inf so they never appear in top-k
        allowed_mask = torch.tensor(
            [mid in allowed_set for mid in self.item_ids],
            device=device,
            dtype=torch.bool,
        )
        if not allowed_mask.any():
            return {"results": [], "debug": {}} if bool(_kwargs.get("return_debug", False)) else []
        scores[~allowed_mask] = float("-inf")
        base_scores[~allowed_mask] = float("-inf")

        sorted_indices = torch.argsort(scores, descending=True)

        results = []
        valid_mask = allowed_mask.clone()
        for idx_t in sorted_indices:
            idx = idx_t.item()
            item_id = self.item_ids[idx]
            if item_id in exclude_set:
                valid_mask[idx] = False
                continue
            results.append(
                {
                    "movie_id": int(item_id),
                    "score": round(scores[idx].item(), 4),
                    "cf_score": round(cf_scores[idx].item(), 4),
                    "genre_score": round(genre_scores[idx].item(), 4),
                    "steering_score": round(steering_scores[idx].item(), 4),
                }
            )
            if len(results) >= n_items:
                break

        if not bool(_kwargs.get("return_debug", False)):
            return results

        movers = self._build_rank_deltas(
            base_scores=base_scores,
            final_scores=scores,
            valid_mask=valid_mask,
            cf_scores=cf_scores,
            genre_scores=genre_scores,
            steering_scores=steering_scores,
            top_k=5,
        )

        # Lightweight influence indicator for user-facing hint.
        valid_steer = steering_scores[valid_mask]
        valid_base = (cf_scores + genre_scores)[valid_mask]
        if int(valid_steer.numel()) > 0 and int(valid_base.numel()) > 0:
            steer_mag = float(torch.quantile(torch.abs(valid_steer), 0.75).item())
            base_span = float(
                (torch.quantile(valid_base, 0.95) - torch.quantile(valid_base, 0.05)).item()
            )
            ratio = steer_mag / max(base_span, 1e-6)
        else:
            ratio = 0.0

        if ratio < 0.08:
            influence_level = "Low impact"
        elif ratio < 0.18:
            influence_level = "Medium impact"
        else:
            influence_level = "High impact"

        return {
            "results": results,
            "debug": {
                "blend_strategy": blend_plan.strategy,
                "cf_weight": round(float(blend_plan.cf_weight), 4),
                "genre_weight": round(float(blend_plan.genre_weight), 4),
                "prior_tiebreak_weight": round(float(blend_plan.prior_tiebreak_weight), 4),
                "adaptive_gamma": round(float(adaptive_gamma), 4),
                "steering_clamp": round(float(clamp_value), 4),
                "steering_ratio": round(float(ratio), 4),
                "influence_level": influence_level,
                "top_up": movers.get("top_up", []),
                "top_down": movers.get("top_down", []),
            },
        }

    def get_item_features(self, item_id: int) -> Dict[str, float]:
        """Get SAE feature activations for a specific item."""
        self.load()

        if self.item_features is None or self.item_ids is None:
            return {}

        try:
            idx = self.item_ids.index(item_id)
        except ValueError:
            return {}

        features = self.item_features[idx]

        result = {}
        topk_values, topk_indices = torch.topk(features, min(10, len(features)))

        for val, feat_idx in zip(topk_values, topk_indices):
            if val.item() > 0:
                label = (self.neuron_labels or {}).get(
                    str(feat_idx.item()),
                    f"Feature {feat_idx.item()}",
                )
                result[label] = round(val.item(), 3)

        return result


_sae_recommenders: Dict[str, SAERecommender] = {}
_default_recommender = None


def get_sae_recommender(model_id: str = None) -> SAERecommender:
    """Get or create SAE recommender instance."""
    global _sae_recommenders, _default_recommender

    if model_id is None:
        if _default_recommender is None:
            _default_recommender = SAERecommender(model_id=DEFAULT_TOPK_SAE_MODEL_ID)
        return _default_recommender

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
    """Convenience function to generate SAE-based recommendations."""
    recommender = get_sae_recommender(model_id=model_id)
    return recommender.get_recommendations(
        feature_adjustments=feature_adjustments,
        n_items=n_items,
        exclude_items=exclude_items,
        allowed_ids=allowed_ids,
    )


def get_available_models() -> List[Dict]:
    """Get list of available SAE models."""
    available = []
    candidates = [
        (DEFAULT_TOPK_SAE_MODEL_ID, "TopKSAE-1024 (k=64)", "Default SAE checkpoint"),
    ]
    for model_id, name, description in candidates:
        path = find_local_model_path(model_id)
        if path is not None:
            available.append(
                {"id": model_id, "name": name, "description": description, "path": str(path)}
            )
    return available
