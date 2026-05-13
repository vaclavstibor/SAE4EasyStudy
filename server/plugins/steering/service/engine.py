"""Shared stateful helpers for steering iterations."""

from __future__ import annotations

import traceback

import numpy as np
from flask import session


def update_elsa_seed_with_likes(
    current_liked_ids: set,
    model_id: str = None,
    like_weight: float = 0.5,
    like_cap: int = 10,
):
    import torch as _torch

    from ..recommendation.sae_recommender import get_sae_recommender

    try:
        recommender = get_sae_recommender(model_id=model_id)
        recommender.load()
        if recommender.item_embeddings is None or recommender.item_ids is None:
            return

        id_to_idx = {int(mid): i for i, mid in enumerate(recommender.item_ids)}
        original_movies = session.get("elicitation_selected_movies", [])

        weighted_sum = np.zeros(recommender.item_embeddings.shape[1], dtype=np.float32)
        total_weight = 0.0

        for mid in original_movies:
            idx = id_to_idx.get(int(mid))
            if idx is not None:
                emb = recommender.item_embeddings[idx]
                if isinstance(emb, _torch.Tensor):
                    emb = emb.cpu().numpy()
                weighted_sum += emb.astype(np.float32)
                total_weight += 1.0

        effective_liked = sorted(int(x) for x in current_liked_ids)[: max(0, int(like_cap))]
        for mid in effective_liked:
            idx = id_to_idx.get(int(mid))
            if idx is not None:
                emb = recommender.item_embeddings[idx]
                if isinstance(emb, _torch.Tensor):
                    emb = emb.cpu().numpy()
                weighted_sum += emb.astype(np.float32) * like_weight
                total_weight += like_weight

        if total_weight > 0:
            new_seed = weighted_sum / total_weight
            session["elsa_seed"] = [round(float(v), 6) for v in new_seed]
            print(
                "[update_elsa_seed_with_likes] Updated seed: "
                f"{len(original_movies)} elicitation + {len(current_liked_ids)} liked "
                f"(effective={len(effective_liked)}, cap={like_cap}, weight={like_weight})"
            )
    except Exception as exc:
        print(f"[update_elsa_seed_with_likes] Error: {exc}")
        traceback.print_exc()

