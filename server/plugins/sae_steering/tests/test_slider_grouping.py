"""Tests for slider clustering, expansion, and sensitivity.

Run from the server/ directory:
    .venv39/bin/python -m pytest plugins/sae_steering/tests/test_slider_grouping.py -v
"""
import pytest

from plugins.sae_steering import (
    _expand_feature_adjustments,
    _select_grouped_slider_features,
    _jaccard_similarity,
    _text_token_set,
    MEMBER_SUPPORT_WEIGHT,
)


# ---------------------------------------------------------------------------
# Unit: clustering merges similar labels, keeps distinct ones separate
# ---------------------------------------------------------------------------

def test_select_grouped_merges_quirky_comedy_variants():
    labeled = [
        {"id": 101, "label": "Quirky Comedy Escapes",
         "description": "Offbeat humor, playful romance, and whimsical comedy energy.",
         "movie_count": 80},
        {"id": 102, "label": "Quirky Comedy Escapades",
         "description": "Whimsical comedy, playful romance, and offbeat humor throughout.",
         "movie_count": 65},
        {"id": 201, "label": "Heroic Quests",
         "description": "Epic journeys of sacrifice and perseverance.",
         "movie_count": 90},
        {"id": 301, "label": "Fast Action Frenzy",
         "description": "High-energy action built around chases and explosive set pieces.",
         "movie_count": 120},
    ]
    score_map = {101: 1.2, 102: 1.1, 201: 0.8, 301: 0.7}

    selected, rejections = _select_grouped_slider_features(
        labeled=labeled, score_map=score_map, decoder_vecs={},
        num_sliders=3, min_neuron_movies=20,
    )

    assert len(selected) == 3
    quirky = next(f for f in selected if f["id"] == 101)
    assert 102 in quirky["member_ids"]
    assert quirky["group_size"] >= 2
    assert "closely related latent features" in quirky["description"]


def test_select_grouped_merges_emotional_family_dramas():
    """Multiple 'Emotional Family Dramas' neurons must collapse into one slider."""
    labeled = [
        {"id": 1492, "label": "Emotional Family Dramas",
         "description": "Intense character-driven stories about family relationships.",
         "movie_count": 400},
        {"id": 3492, "label": "Emotional Family Dramas",
         "description": "Intense character-driven stories about family dynamics.",
         "movie_count": 447},
        {"id": 7082, "label": "Emotional Family Dramas",
         "description": "Intense character-driven stories about family dynamics and love.",
         "movie_count": 419},
        {"id": 5000, "label": "Fast-Paced Action Thrills",
         "description": "Non-stop chases, explosions, and adrenaline-pumping stunts.",
         "movie_count": 200},
    ]
    score_map = {1492: 1.0, 3492: 0.9, 7082: 0.85, 5000: 0.7}

    selected, _ = _select_grouped_slider_features(
        labeled=labeled, score_map=score_map, decoder_vecs={},
        num_sliders=10, min_neuron_movies=20,
    )

    family_sliders = [f for f in selected if "family" in f["label"].lower()]
    assert len(family_sliders) == 1, (
        f"Expected 1 family slider, got {len(family_sliders)}: "
        f"{[f['label'] for f in family_sliders]}"
    )
    assert set(family_sliders[0]["member_ids"]) == {1492, 3492, 7082}


def test_select_grouped_merges_emotional_character_and_turmoil():
    """'Emotional Character Studies' and 'Emotional Turmoil Movies' should merge."""
    labeled = [
        {"id": 7095, "label": "Emotional Character Studies",
         "description": "Intense character-driven dramas about human emotions, flawed protagonists.",
         "movie_count": 81},
        {"id": 2628, "label": "Emotional Turmoil Movies",
         "description": "Intense character-driven stories about human emotions, struggle and sacrifice.",
         "movie_count": 4500},
        {"id": 4833, "label": "Fast-Paced Action Thrills",
         "description": "Explosive action sequences, chases, and adrenaline-pumping stunts.",
         "movie_count": 128},
    ]
    score_map = {7095: 1.3, 2628: 1.1, 4833: 0.9}

    selected, _ = _select_grouped_slider_features(
        labeled=labeled, score_map=score_map, decoder_vecs={},
        num_sliders=10, min_neuron_movies=20,
    )

    emotional_sliders = [f for f in selected if "emotional" in f["label"].lower()]
    assert len(emotional_sliders) == 1, (
        f"Expected 1 emotional slider, got {len(emotional_sliders)}: "
        f"{[f['label'] for f in emotional_sliders]}"
    )
    assert 7095 in emotional_sliders[0]["member_ids"]
    assert 2628 in emotional_sliders[0]["member_ids"]


def test_too_broad_neurons_rejected():
    """Neurons with >5000 movies should be filtered as too broad."""
    labeled = [
        {"id": 7174, "label": "Emotional Family Dramas",
         "description": "Stories about family.",
         "movie_count": 36012},
        {"id": 100, "label": "Specific Action",
         "description": "Very specific action concept.",
         "movie_count": 200},
    ]
    score_map = {7174: 1.5, 100: 0.8}

    selected, rejections = _select_grouped_slider_features(
        labeled=labeled, score_map=score_map, decoder_vecs={},
        num_sliders=10, min_neuron_movies=20,
    )

    assert len(selected) == 1
    assert selected[0]["id"] == 100
    rejected_ids = [r[0] for r in rejections if r[2] == "too_broad"]
    assert 7174 in rejected_ids


def test_distinct_concepts_stay_separate():
    labeled = [
        {"id": 1, "label": "Dark Psychological Thrills",
         "description": "Suspenseful mind-bending narratives with unreliable narrators.",
         "movie_count": 100},
        {"id": 2, "label": "Wholesome Family Fun",
         "description": "Warm nostalgic tales celebrating childhood and music.",
         "movie_count": 66},
        {"id": 3, "label": "Fast-Paced Action Thrills",
         "description": "Explosive action, car chases, and martial arts.",
         "movie_count": 128},
    ]
    score_map = {1: 1.0, 2: 0.9, 3: 0.8}

    selected, _ = _select_grouped_slider_features(
        labeled=labeled, score_map=score_map, decoder_vecs={},
        num_sliders=10, min_neuron_movies=20,
    )

    assert len(selected) == 3
    for f in selected:
        assert f["group_size"] == 1


# ---------------------------------------------------------------------------
# Unit: diversity -- no two final sliders share Jaccard > 0.5 on label tokens
# ---------------------------------------------------------------------------

def test_diversity_no_high_jaccard_pairs():
    """Top-14 sliders must have pairwise label Jaccard < 0.5."""
    labeled = []
    for i in range(40):
        if i < 10:
            labeled.append({
                "id": 1000 + i,
                "label": f"Emotional Family Dramas V{i}",
                "description": "Intense character-driven stories about family.",
                "movie_count": 300 - i * 10,
            })
        elif i < 20:
            labeled.append({
                "id": 2000 + i,
                "label": f"Quirky Comedy Variant {i}",
                "description": "Offbeat humor and whimsical storytelling.",
                "movie_count": 200 - i * 5,
            })
        else:
            labeled.append({
                "id": 3000 + i,
                "label": f"Unique Concept {i}",
                "description": f"A totally unique concept number {i}.",
                "movie_count": 100,
            })

    score_map = {f["id"]: 2.0 - i * 0.04 for i, f in enumerate(labeled)}

    selected, _ = _select_grouped_slider_features(
        labeled=labeled, score_map=score_map, decoder_vecs={},
        num_sliders=14, min_neuron_movies=20,
    )

    for i, a in enumerate(selected):
        for j, b in enumerate(selected):
            if i >= j:
                continue
            tokens_a = _text_token_set(a["label"])
            tokens_b = _text_token_set(b["label"])
            jac = _jaccard_similarity(tokens_a, tokens_b)
            assert jac < 0.50, (
                f"Sliders [{a['label']}] and [{b['label']}] have Jaccard={jac:.2f} >= 0.50"
            )


# ---------------------------------------------------------------------------
# Unit: anchor-gets-full-delta expansion
# ---------------------------------------------------------------------------

def test_expand_anchor_gets_full_delta():
    """Anchor neuron should receive the full slider delta."""
    current_features = [
        {
            "id": 100,
            "label": "Emotional",
            "member_ids": [100, 200, 300],
        },
    ]

    expanded = _expand_feature_adjustments(
        raw_adjustments={"100": 1.0},
        current_features=current_features,
        cluster_map={},
    )

    assert abs(expanded["100"] - 1.0) < 1e-6, "Anchor should get full delta"
    assert abs(expanded["200"] - MEMBER_SUPPORT_WEIGHT) < 1e-6
    assert abs(expanded["300"] - MEMBER_SUPPORT_WEIGHT) < 1e-6


def test_expand_single_member_gets_full_delta():
    """Single-neuron slider should receive the full delta."""
    current_features = [
        {"id": 201, "label": "Heroic Quests"},
    ]

    expanded = _expand_feature_adjustments(
        raw_adjustments={"201": -0.75},
        current_features=current_features,
        cluster_map={},
    )

    assert abs(expanded["201"] - (-0.75)) < 1e-6


def test_expand_cluster_adjustments():
    """Cluster-level adjustments propagate to all neurons."""
    expanded = _expand_feature_adjustments(
        raw_adjustments={"cluster_action": 0.5},
        current_features=[],
        cluster_map={"cluster_action": [10, 20, 30]},
    )

    assert expanded == {"10": 0.5, "20": 0.5, "30": 0.5}


def test_expand_mixed_slider_and_cluster():
    current_features = [
        {
            "id": 101,
            "label": "Quirky",
            "member_ids": [101, 102],
        },
        {"id": 201, "label": "Heroic"},
    ]
    cluster_map = {"cluster_story": [301, 302]}

    expanded = _expand_feature_adjustments(
        raw_adjustments={"101": 0.5, "201": -0.25, "cluster_story": 0.2},
        current_features=current_features,
        cluster_map=cluster_map,
    )

    assert abs(expanded["101"] - 0.5) < 1e-6
    assert abs(expanded["102"] - 0.5 * MEMBER_SUPPORT_WEIGHT) < 1e-6
    assert abs(expanded["201"] - (-0.25)) < 1e-6
    assert abs(expanded["301"] - 0.2) < 1e-6
    assert abs(expanded["302"] - 0.2) < 1e-6
