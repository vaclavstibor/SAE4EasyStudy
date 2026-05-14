"""Recommendation generation and payload formatting."""

import os
import traceback

import numpy as np
from flask import session

from ..constants import DEFAULT_TOPK_SAE_MODEL_ID

TMDB_CACHE = None


def load_tmdb_overviews():
    global TMDB_CACHE
    if TMDB_CACHE is not None:
        return TMDB_CACHE
    plots_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "static",
        "datasets",
        "ml-32m-filtered",
        "plots.csv",
    )
    TMDB_CACHE = {}
    if os.path.exists(plots_path):
        try:
            import csv

            with open(plots_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    movie_id = row.get("movieId", "")
                    plot = row.get("plot", "")
                    if movie_id and plot:
                        TMDB_CACHE[int(movie_id)] = plot
            print(f"[Plots] Loaded {len(TMDB_CACHE)} movie plots")
        except Exception as exc:
            print(f"[Plots] Could not load plots: {exc}")
    return TMDB_CACHE


def compute_genre_bonus(recommender, loader, seed_genres: set) -> np.ndarray:
    n_items = len(recommender.item_ids)
    bonus = np.zeros(n_items, dtype=np.float32)
    if not seed_genres:
        return bonus
    for i, mid in enumerate(recommender.item_ids):
        mid = int(mid)
        try:
            row = loader.movies_df_indexed.loc[mid]
            item_genres = {
                genre.strip()
                for genre in str(row.genres).split("|")
                if genre.strip() and genre.strip() != "(no genres listed)"
            }
            if item_genres:
                overlap = len(item_genres & seed_genres)
                union = len(item_genres | seed_genres)
                bonus[i] = overlap / union
        except (KeyError, AttributeError):
            pass
    return bonus


def unwrap_recommendation_payload(payload):
    if isinstance(payload, dict):
        return payload.get("recommendations", []), payload.get("debug", {})
    return payload or [], {}


def generate_steered_recommendations_for_model(
    loader,
    selected_movies,
    feature_adjustments,
    model_config,
    k=20,
    suppressed_genres=None,
    reranking_strategy=None,
    reranking_params=None,
):
    from .sae_recommender import get_sae_recommender

    suppressed_genres = suppressed_genres or []
    sae_model_id = model_config.get("sae", DEFAULT_TOPK_SAE_MODEL_ID)
    try:
        recommender = get_sae_recommender(model_id=sae_model_id)
        recommender.load()
        if recommender.item_features is None or recommender.item_ids is None:
            print(
                "[generate_steered_recs] SAE runtime activations missing; "
                "falling back to metadata-based recommendations"
            )
            return fallback_genre_recommendations(loader, selected_movies, feature_adjustments, k)

        neuron_adjustments = {int(key): float(value) for key, value in feature_adjustments.items()}
        n_adj = sum(1 for value in neuron_adjustments.values() if abs(value) > 0.001)
        print(f"[generate_steered_recs] SAE model={sae_model_id}, non-zero adjustments={n_adj}")

        exclude_movie_ids = []
        for movie_ref in selected_movies:
            try:
                exclude_movie_ids.append(int(movie_ref))
            except (ValueError, TypeError):
                continue
        allowed_ids = set(loader.movies_df_indexed.index.tolist())

        elsa_seed_list = session.get("elsa_seed")
        elsa_seed = np.array(elsa_seed_list, dtype=np.float32) if elsa_seed_list else None
        seed_genres = set(session.get("seed_genres", []))
        genre_bonus = compute_genre_bonus(recommender, loader, seed_genres) if seed_genres else None

        if elsa_seed is not None:
            print(f"[generate_steered_recs] ELSA seed active, genres={seed_genres}")

        rec_payload = recommender.get_recommendations(
            feature_adjustments=neuron_adjustments,
            n_items=max(k * 15, 300),
            exclude_items=exclude_movie_ids,
            allowed_ids=allowed_ids,
            seed_embedding=elsa_seed,
            genre_bonus=genre_bonus,
            return_debug=True,
            reranking_strategy=reranking_strategy,
            reranking_params=reranking_params,
        )
        raw_recommendations = (
            rec_payload.get("results", []) if isinstance(rec_payload, dict) else rec_payload
        )
        debug_payload = rec_payload.get("debug", {}) if isinstance(rec_payload, dict) else {}
        print(f"[generate_steered_recs] Raw recommendations: {len(raw_recommendations)}")
        if debug_payload:
            print(
                "[generate_steered_recs] influence="
                f"{debug_payload.get('influence_level')} "
                f"(gamma={debug_payload.get('adaptive_gamma')}, "
                f"clamp={debug_payload.get('steering_clamp')}, "
                f"ratio={debug_payload.get('steering_ratio')})"
            )

        overviews = load_tmdb_overviews()
        results = []
        skipped_missing_meta = []
        skipped_unknown_id = []
        for rec in raw_recommendations:
            movie_id = rec.get("movie_id")
            title = None
            genres = []
            image_url = None

            if movie_id not in loader.movies_df_indexed.index:
                skipped_unknown_id.append(movie_id)
                continue

            try:
                movie_info = loader.movies_df_indexed.loc[movie_id]
                title = movie_info.title
                genres = movie_info.genres.split("|")
                try:
                    movie_idx = loader.movie_id_to_index[movie_id]
                    image_url = loader.get_image(movie_idx)
                except (KeyError, AttributeError):
                    image_url = None
            except (KeyError, AttributeError):
                try:
                    fallback_row = loader.movies_df[loader.movies_df.movieId == movie_id]
                    if not fallback_row.empty:
                        title = fallback_row.iloc[0].title
                        genres = fallback_row.iloc[0].genres.split("|")
                except Exception:
                    pass

            if not title:
                skipped_missing_meta.append(movie_id)
                continue

            if suppressed_genres and any(sg in genres for sg in suppressed_genres):
                continue

            results.append(
                {
                    "title": title,
                    "movie_idx": movie_id,
                    "score": rec.get("score", 0.5),
                    "metadata": " | ".join(
                        [genre for genre in genres if genre != "(no genres listed)"]
                    ),
                    "matched_features": rec.get("matched_features", {}),
                    "model": model_config.get("id", "unknown"),
                    "url": image_url,
                    "overview": overviews.get(movie_id, ""),
                }
            )
            if len(results) >= k:
                break

        if skipped_unknown_id:
            print(
                "[generate_steered_recommendations_for_model] "
                f"Skipped {len(skipped_unknown_id)} items with unknown IDs, "
                f"sample: {skipped_unknown_id[:10]}"
            )
        if skipped_missing_meta:
            print(
                "[generate_steered_recommendations_for_model] "
                f"Skipped {len(skipped_missing_meta)} items with missing metadata, "
                f"sample: {skipped_missing_meta[:10]}"
            )
        if len(results) < k:
            print(
                "[generate_steered_recommendations_for_model] WARNING: "
                f"only {len(results)} results after filtering (target {k})"
            )
        print(
            "[generate_steered_recommendations_for_model] "
            f"Returning {len(results)} recommendations (target {k})"
        )
        return {"recommendations": results[:k], "debug": debug_payload}
    except Exception as exc:
        print(f"[generate_steered_recommendations_for_model] Error: {exc}")
        traceback.print_exc()
        return fallback_genre_recommendations(loader, selected_movies, feature_adjustments, k)


def generate_steered_recommendations(loader, selected_movies, feature_adjustments, k=20):
    payload = generate_steered_recommendations_for_model(
        loader=loader,
        selected_movies=selected_movies,
        feature_adjustments=feature_adjustments,
        model_config={"sae": DEFAULT_TOPK_SAE_MODEL_ID},
        k=k,
    )
    if isinstance(payload, dict):
        return payload.get("recommendations", [])
    return payload


def fallback_genre_recommendations(loader, selected_movies, feature_adjustments, k=20):
    feature_to_genres = {
        "0": ["Action", "Adventure"],
        "1": ["Drama"],
        "2": ["Comedy"],
        "3": ["Sci-Fi", "Fantasy"],
        "4": ["Thriller", "Mystery"],
        "5": ["Romance"],
        "6": ["Horror"],
        "7": ["Animation", "Children"],
        "8": ["Documentary"],
        "9": ["War", "Western", "Film-Noir"],
    }

    genre_weights = {}
    for feature_id, adjustment in feature_adjustments.items():
        for genre in feature_to_genres.get(str(feature_id), []):
            genre_weights[genre] = genre_weights.get(genre, 1.0) * float(adjustment)

    candidate_indices = [
        idx for idx in loader.movie_index_to_id.keys() if idx not in selected_movies
    ]
    scored_movies = []
    for movie_idx in candidate_indices[:500]:
        try:
            movie_id = loader.movie_index_to_id[movie_idx]
            movie_genres = loader.movies_df_indexed.loc[movie_id].genres.split("|")
            title = loader.movies_df_indexed.loc[movie_id].title
            genre_score = 0.0
            for genre in movie_genres:
                if genre in genre_weights:
                    genre_score += genre_weights[genre] - 1.0
            final_score = max(0.0, min(1.0, 0.5 + genre_score * 0.3))
            try:
                image_url = loader.get_image(movie_idx)
            except (KeyError, AttributeError):
                image_url = None
            scored_movies.append(
                {
                    "title": title,
                    "movie_idx": movie_id,
                    "score": final_score,
                    "metadata": " | ".join(
                        [genre for genre in movie_genres if genre != "(no genres listed)"]
                    ),
                    "url": image_url,
                }
            )
        except (KeyError, AttributeError):
            continue
    scored_movies.sort(key=lambda x: -x["score"])
    return scored_movies[:k]
