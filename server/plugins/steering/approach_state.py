"""Helpers for approach-scoped session state.

The persisted session keys still use historical ``*_per_phase`` names so
in-progress local studies and result exports remain readable.
"""

from flask import session


def get_approach_id_map(session_key: str) -> dict:
    raw = session.get(session_key, {})
    if isinstance(raw, dict):
        return raw
    return {}


def get_approach_movie_set(session_key: str, approach_idx: int) -> set:
    approach_map = get_approach_id_map(session_key)
    raw_list = approach_map.get(str(int(approach_idx)), [])
    if not isinstance(raw_list, list):
        return set()
    return {int(mid) for mid in raw_list if mid is not None}


def set_approach_movie_set(session_key: str, approach_idx: int, movie_ids: set) -> None:
    approach_map = get_approach_id_map(session_key)
    approach_map[str(int(approach_idx))] = sorted(
        {int(mid) for mid in movie_ids if mid is not None}
    )
    session[session_key] = approach_map


def get_approach_token_set(session_key: str, approach_idx: int) -> set:
    approach_map = get_approach_id_map(session_key)
    raw_list = approach_map.get(str(int(approach_idx)), [])
    if not isinstance(raw_list, list):
        return set()
    return {str(token) for token in raw_list if token is not None}


def set_approach_token_set(session_key: str, approach_idx: int, tokens: set) -> None:
    approach_map = get_approach_id_map(session_key)
    approach_map[str(int(approach_idx))] = sorted(
        {str(token) for token in tokens if token is not None}
    )
    session[session_key] = approach_map


def remember_shown_movies(approach_idx: int, movie_ids: list) -> None:
    if not movie_ids:
        return
    seen = get_approach_movie_set("seen_movies_per_phase", approach_idx)
    seen.update({int(mid) for mid in movie_ids if mid is not None})
    set_approach_movie_set("seen_movies_per_phase", approach_idx, seen)
