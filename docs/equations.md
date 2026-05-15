# Equations and Scoring

This page is the math reference for the SAE Steering plugin. Each modality follows the same shape: **Input · Per-iteration weight · Composition · Code**. Every formula is paired with a short code snippet so the math and the implementation can be checked side by side.

**Notation rules used throughout this page.**

- Single-letter symbols ($w$, $\delta_c$, $\alpha$, $\beta$, $\gamma$, $\tau$, $\lambda$) are reserved for math.
- Multi-letter names that refer to *functions or fields* are typeset with `\mathrm{...}` (e.g. $\mathrm{score}(s, c)$, $\mathrm{cf}(i)$). This keeps `direction(s)` from being rendered as $d \cdot i \cdot r \cdot e \cdot c \cdot t \cdot \dots$
- Implementation identifiers — Python variable names, config keys, session keys, files — are rendered as `code`.
- Every numeric constant in a formula is either a **config key** (researcher-tunable; see the Appendix) or **hardcoded**. Inline math sections only use the symbol; the Appendix is the single source of truth for values and scopes.

**See also.**

- [`tech-docs.md` Section 6](tech-docs.md#6-steering-modalities-and-the-iteration-loop) — the iteration loop that consumes these weights.
- [`design-decisions.md` Section 6](design-decisions.md#6-text-steering-composition-is-a-configurable-mode-fr-09), [Section 7](design-decisions.md#7-nfr-12-text-steering-ambiguity-degrades-gracefully), [Section 8](design-decisions.md#8-reranking-strategy-as-a-typed-enum-fr-10--schema-and-dispatch-contract) — rationale for composition modes, the no-match fallback, and the reranking enum.
- [`formative-examples.md` Section 3](formative-examples.md#3-add-a-new-dataset), [Section 5](formative-examples.md#5-add-a-new-reranking-strategy) — how to add a new modality / a new reranking strategy.

## Contents

### Steering modalities

1. [Common framework](#1-common-framework)  
   The shared modality interface and cluster→neuron expansion.
2. [Sliders (FR-05/06)](#2-sliders-fr-0506)  
   Slider deltas, amplification, and per-iteration accumulation.
3. [Toggles (FR-07)](#3-toggles-fr-07)  
   Signed fixed-magnitude weights.
4. [Text steering (FR-09)](#4-text-steering-fr-09)  
   Segmentation, token/phrase scoring, top-K, and composition modes.
5. [Example-based steering (FR-08)](#5-example-based-steering-fr-08)  
   Mean activation over liked examples and top-K selection.
6. [Reset (FR-12)](#6-reset-fr-12)  
   Session clearing + audit contract.

### Base recommender and reranking

7. [ELSA seed re-weighting from likes](#7-elsa-seed-re-weighting-from-likes)  
   How likes bias the seed embedding.
8. [Final ranking — cross-strategy summary](#8-final-ranking--cross-strategy-summary)  
   How each strategy combines the three building blocks.
9. [Notation summary](#9-notation-summary)  
   Symbol table used across equations.
10. [Reranking strategies](#10-reranking-strategies-reranking_strategy-config-key)  
    `feature-conditioned`, `latent-perturbation`, `constrained-subset`.

### Reference

- [Appendix: What is configurable](#appendix-what-is-configurable)  
  Config keys, defaults, and scopes.

## 1. Common framework

All four user-facing modalities in `server/plugins/steering/modalities/` implement the same strategy interface:

```16:22:server/plugins/steering/modalities/base.py
class SteeringModality:
    """Small strategy interface for one user-facing steering modality."""

    modality_id: str = ""

    def apply(self, data: Dict[str, Any], *, conf: dict, active_model: dict) -> SteeringResult:
        raise NotImplementedError
```

A `SteeringResult` is a triple `(features, adjustments, metadata)`. The `adjustments` field is a dictionary from cluster id to a real weight:

$$
\mathrm{adjustments} : c \mapsto w(c), \qquad w(c) \in [-1, 1].
$$

Semantics: $w(c) > 0$ means **boost** items whose neurons fire in cluster $c$; $w(c) < 0$ means **suppress** them.

### 1.1 Cluster-to-neuron expansion

The recommender ranks items by an inner product against a per-**neuron** profile, not a per-cluster one. The bridge is `expand_feature_adjustments`: each cluster-level value $\delta_c$ is fanned out to every neuron $n$ in that cluster's neuron set $M[c]$; overlapping clusters sum:

$$
\Delta_n \;=\; \sum_{c : n \in M[c]} \delta_c
$$

Adjustments with $|\delta_c| < 10^{-4}$ are dropped before the expansion.

```38:52:server/plugins/steering/recommendation/semantic_registry.py
def expand_feature_adjustments(raw_adjustments: dict, cluster_map: dict = None) -> dict:
    feature_adjustments = {}
    cluster_map = cluster_map or {}
    for key, val in (raw_adjustments or {}).items():
        delta = float(val)
        if abs(delta) < 0.0001:
            continue
        neuron_ids = cluster_map.get(key)
        if neuron_ids:
            for nid in neuron_ids:
                skey = str(nid)
                feature_adjustments[skey] = feature_adjustments.get(skey, 0.0) + delta
        else:
            feature_adjustments[key] = feature_adjustments.get(key, 0.0) + delta
    return feature_adjustments
```

### 1.2 How the expanded vector enters ranking

Each item $i$ has a row $f_i \in \mathbb{R}^n$ of SAE activations (where $n$ is the SAE feature count, e.g. $1024$ for `TopKSAE-1024`). The recommender builds a sparse profile $a \in \mathbb{R}^n$ with $a_n = \Delta_n$ (after expansion), and computes the SAE score as the inner product:

$$
\mathrm{sae}(i) \;=\; f_i^\top a.
$$

Implemented as a matrix-vector product over the whole catalogue:

```476:485:server/plugins/steering/recommendation/sae_recommender.py
        sae_scores = (
            torch.matmul(self.item_features, sae_profile)
            if has_adjustments
            else torch.zeros(n_items_total, device=device)
        )
```

The full final score (CF + genre + the strategy-specific SAE contribution + a small tiebreak term) is in [Section 8](#8-final-ranking--cross-strategy-summary) and [Section 10](#10-reranking-strategies-reranking_strategy-config-key); the per-modality sections below cover only how each modality produces its slice of $\delta_c$.

## 2. Sliders (FR-05/06)

#### Input

The UI sends a dictionary $\mathrm{raw}: c \mapsto \delta_c \in [-1, 1]$, one slider per visible cluster.

#### Per-iteration weight

`SliderSteering` multiplies each non-trivial value by a fixed amplification $\alpha = 2.0$ (`SLIDER_AMPLIFICATION`, hardcoded):

$$
w_{\mathrm{slider}}(c) \;=\; \alpha \cdot \delta_c, \qquad |\delta_c| > 10^{-3}.
$$

Values with $|\delta_c| \le 10^{-3}$ are discarded.

```16:28:server/plugins/steering/modalities/sliders.py
class SliderSteering(SteeringModality):
    modality_id = Modalities.SLIDERS

    def apply(self, data: Dict[str, Any], *, conf: dict, active_model: dict) -> SteeringResult:
        raw_adjustments = data.get("adjustments", {}) or {}
        adjustments = {
            str(feature_id): round(float(value) * SLIDER_AMPLIFICATION, 4)
            for feature_id, value in raw_adjustments.items()
            if abs(float(value)) > 0.001
        }
        return SteeringResult(
            features=[], adjustments=adjustments, metadata={"raw_adjustments": raw_adjustments}
        )
```

#### Composition across iterations

Sliders **accumulate**, not replace. The iteration controller keeps a `previous_adjustments` map from neuron id to weight in the session, and after expanding the cluster-level deltas to neurons, adds the amplified increment per neuron:

$$
\Delta_n^{(t)} \;=\; \Delta_n^{(t-1)} \,+\, \alpha \cdot \delta_n^{(t)}.
$$

```161:170:server/plugins/steering/service/iteration_controller.py
    for key, val in feature_adjustments.items():
        skey = str(key)
        prev = float(previous_adjustments.get(skey, 0))
        raw_delta = float(val)
        new = raw_delta * SLIDER_AMPLIFICATION
        if abs(raw_delta) > 0.001:
            previous_adjustments[skey] = round(prev + new, 4)
        elif skey in previous_adjustments and abs(prev) < 0.001:
            del previous_adjustments[skey]
```

The cumulative neuron map is persisted in `session["cumulative_adjustments"]`. A subsequent slider refresh (`compute_updated_sliders`) hides clusters already touched in this phase so the participant always sees fresh content to steer with — touched clusters are filtered out of the next pool, **not** erased from the cumulative map.

### 2.1 Interaction-history mode (`cumulative` vs `reset`)

The study-level `interaction_mode` config key (default `cumulative`, surfaced in the admin UI as "Interaction History Mode") controls whether the per-iteration steering memory survives into the next iteration of the *same* approach.

| Mode | What happens at the top of iteration $t+1$ |
| --- | --- |
| `cumulative` | `previous_adjustments` is loaded from `session["cumulative_adjustments"]`; new deltas accumulate on top per the formula above; `session["user_touched_features"]` grows monotonically until the approach ends; `update_elsa_seed_with_likes(L, …)` keeps the ELSA seed re-weighted by the current liked set (see [Section 7](#7-elsa-seed-re-weighting-from-likes)). |
| `reset` | `cumulative_adjustments`, `user_touched_features`, `last_text_steering`, `last_example_steering`, `excluded_movies_from_text` and `boosted_liked_ids` are cleared; `update_elsa_seed_with_likes(set(), …)` rebuilds the seed from `elicitation_selected_movies` alone (no like re-weighting). Audit rows still capture the current iteration's adjustments and liked set. |

Switching between approaches always clears the per-approach state (`_do_advance_phase` in `routes/study.py`), regardless of the mode, so no active steering signal can leak from one approach into the next. The text-steering scope guard described in [Section 4.5](#45-top-k-and-composition-across-iterations) also rejects any residual entry whose `(study_guid, phase)` does not match the current one.

## 3. Toggles (FR-07)

#### Input

The UI sends $\mathrm{raw} : c \mapsto \delta_c \in \mathbb{R}$, where only the **sign** is meaningful: positive means boost, negative means suppress, and near-zero means unset.

#### Per-iteration weight

Each touched cluster gets a fixed-magnitude weight $\beta$ (`toggle_default_weight`; default `0.65`), signed by the user's choice:

$$
w_{\mathrm{toggle}}(c) \;=\; \mathrm{sign}(\delta_c) \cdot \beta, \qquad |\delta_c| > 10^{-3}.
$$

```13:32:server/plugins/steering/modalities/toggles.py
class ToggleSteering(SteeringModality):
    modality_id = Modalities.TOGGLES

    def apply(self, data: Dict[str, Any], *, conf: dict, active_model: dict) -> SteeringResult:
        raw_adjustments = data.get("adjustments", {}) or {}
        default_weight = float(
            active_model.get(
                "toggle_default_weight", conf.get("toggle_default_weight", DEFAULT_TOGGLE_WEIGHT)
            )
        )
        adjustments = {}
        for feature_id, value in raw_adjustments.items():
            numeric = float(value)
            if abs(numeric) <= 0.001:
                continue
            sign = 1.0 if numeric > 0 else -1.0
            adjustments[str(feature_id)] = round(sign * default_weight, 4)
```

#### Composition across iterations

Toggles share the slider accumulation path (the iteration controller treats every post-expansion neuron delta uniformly; the slider amplification $\alpha = 2$ is applied again at the neuron step). The practical consequence is that two consecutive *boost* actions on the same cluster **reinforce** it, while a *boost* followed by a *suppress* mostly cancels out.

## 4. Text steering (FR-09)

#### Input

A free-form participant query $Q$, with $|Q| \le \mathrm{max\_chars}$ (`text_steering.max_query_chars`, study-level; default `200`). The route returns HTTP 400 if $Q$ is longer.

### 4.1 Segmentation

$Q$ is split on the regex `_SEGMENT_BOUNDARY_RE = [.;]|\bbut\b|\bhowever\b`. Each non-empty chunk becomes a segment $s_i$ with three derived quantities:

| Quantity | Domain | Definition |
| --- | --- | --- |
| $\mathrm{direction}(s_i)$ | $\{+1, -1\}$ | $-1$ iff any marker in `_NEGATIVE_HINTS` (`not`, `no`, `never`, `don't`, `i hate`, …) appears in $s_i$; otherwise $+1$. |
| $\mathrm{intensity}(s_i)$ | $\{0.65, 1.0, 1.35\}$ | $1.35$ for "much/way/a lot more|less", "strongly", "definitely"; $0.65$ for "slightly", "a bit", "somewhat", "kind of"; $1.0$ otherwise. |
| $\mathrm{tokens}(s_i)$ | finite multiset of strings | Alphanumeric tokens of length $\ge 2$, lower-cased, with `_STOP_WORDS` removed. |

The hint lists, segmentation regex, intensity ladder, and stop-word list are **hardcoded** in `modalities/text.py`.

```102:118:server/plugins/steering/modalities/text.py
def _split_query(query: str) -> List[Dict]:
    chunks = _SEGMENT_BOUNDARY_RE.split(query or "")
    segments = []
    for chunk in chunks:
        text = chunk.strip()
        if not text:
            continue
        lowered = text.lower()
        direction = -1 if any(marker in lowered for marker in _NEGATIVE_HINTS) else 1
        segments.append(
            {
                "text": text,
                "direction": direction,
                "intensity": _intensity_multiplier(lowered),
                "tokens": _tokenize(text),
            }
        )
```

### 4.2 Cluster scoring

For each cluster $c$ with label $\ell_c$ and description $d_c$, and each segment $s_i$, define the haystack $h_c = \ell_c \cup d_c$ (whitespace-joined). The per-segment score is

$$
\mathrm{score}(s_i, c) \;=\; 3 \cdot \mathbb{1}[\mathrm{text}(s_i) \subseteq h_c] \;+\; \sum_{t \in \mathrm{tokens}(s_i)} w(t, c) \;+\; \frac{|\mathrm{tokens}(s_i) \cap \mathrm{tokens}(h_c)|}{|\mathrm{tokens}(s_i)|},
$$

with per-token weight

$$
w(t, c) \;=\;
\begin{cases}
2.5  & t \in \mathrm{tokens}(\ell_c), \\
1.25 & t \in \mathrm{tokens}(d_c), \\
0.75 & t \in h_c \text{ but no token match}, \\
0    & \text{otherwise.}
\end{cases}
$$

```132:156:server/plugins/steering/modalities/text.py
def _score_cluster(segment: Dict, cluster: Dict) -> float:
    label = (cluster.get("label") or "").lower()
    description = (cluster.get("description") or "").lower()
    haystack = f"{label} {description}".strip()
    if not haystack:
        return 0.0
    score = 0.0
    phrase = segment.get("text", "").lower().strip()
    if phrase and phrase in haystack:
        score += 3.0
    tokens = segment.get("tokens") or []
    if not tokens:
        return score
    label_tokens = set(_tokenize(label))
    desc_tokens = set(_tokenize(description))
    for token in tokens:
        if token in label_tokens:
            score += 2.5
        elif token in desc_tokens:
            score += 1.25
        elif token in haystack:
            score += 0.75
    coverage = len([token for token in tokens if token in haystack]) / max(len(tokens), 1)
    score += coverage
    return score
```

### 4.3 Aggregation across segments

Let $S_c = \{ i : \mathrm{score}(s_i, c) > 0 \}$ be the segments that contribute to cluster $c$. Then

$$
\mathrm{total}(c) \;=\; \sum_{i \in S_c} \mathrm{score}(s_i, c),
$$

$$
\overline{\mathrm{dir}}(c) \;=\;
\begin{cases}
+1 & \text{if } \sum_{i \in S_c} \mathrm{direction}(s_i) \ge 0, \\
-1 & \text{otherwise,}
\end{cases}
\qquad
\overline{\mathrm{int}}(c) \;=\; \frac{1}{|S_c|} \sum_{i \in S_c} \mathrm{intensity}(s_i).
$$

### 4.4 Weight assignment

With baseline $w^\ast =$ `text_steering_weight` (default `0.55`), bounds $[0.25, 0.95]$ and $[0.1, 0.95]$ hardcoded:

$$
w_0(c) \;=\; \mathrm{clip}\!\left( w^\ast + \tfrac{\mathrm{total}(c)}{10},\; 0.25,\; 0.95 \right),
$$

$$
w(c) \;=\; \overline{\mathrm{dir}}(c) \cdot \mathrm{clip}\!\left( w_0(c) \cdot \overline{\mathrm{int}}(c),\; 0.1,\; 0.95 \right).
$$

```186:202:server/plugins/steering/modalities/text.py
        avg_direction = 1 if not direction_votes else (1 if sum(direction_votes) >= 0 else -1)
        avg_intensity = (intensity_sum / contributing_segments) if contributing_segments else 1.0
        normalized_weight = min(0.95, max(0.25, default_weight + (total_score / 10.0)))
        normalized_weight = min(0.95, max(0.1, normalized_weight * avg_intensity))
        cluster_id = cluster.get("cluster_id") or cluster.get("id")
        scored.append(
            {
                "id": cluster_id,
                "label": cluster.get("label") or str(cluster_id),
                "description": cluster.get("description", ""),
                "weight": round(normalized_weight * avg_direction, 2),
                "direction": "boost" if avg_direction >= 0 else "suppress",
                "match_score": round(total_score, 3),
                "intensity": round(avg_intensity, 2),
                "member_ids": cluster.get("member_ids") or cluster.get("neuron_ids") or [],
            }
        )
```

### 4.5 `top-K` and composition across iterations

The top `text_steering_top_k` clusters (default `6`) by $|w(c)|$ survive; the others are dropped. Their $(c, w(c))$ map is the iteration's text adjustments — call it $T_t$ at iteration $t$.

Composition with the previous iteration's map $T_{t-1}$ is governed by the active approach's `text_composition_mode` (study-level fallback `text_steering.composition_mode`). The three supported modes:

#### 4.5.1 `replace` (default)

$$
T_t^{\mathrm{eff}} \;=\; T_t.
$$

The current prompt's weights replace any previous map entirely. Previous text adjustments are **discarded**, regardless of which clusters they targeted. This is the right default when each prompt expresses a fresh request.

| Cluster | $T_{t-1}$ | $T_t$ | $T_t^{\mathrm{eff}}$ |
| --- | --- | --- | --- |
| $c_{17}$ | $+0.6$ | — | — |
| $c_{42}$ | $-0.4$ | $+0.5$ | $+0.5$ |

Last prompt was *"more sci-fi, less romance"*; the new prompt is *"more romance"*. The sci-fi boost is gone, the romance direction has flipped to positive.

#### 4.5.2 `add`

$$
T_t^{\mathrm{eff}}(c) \;=\; \mathrm{clip}\!\bigl( T_{t-1}(c) + T_t(c),\; -0.95,\; +0.95 \bigr), \qquad T_{t-1}(c) = 0 \text{ for } c \notin T_{t-1}.
$$

Per-cluster weights from the two iterations are summed. Clusters that appear in only one of $T_{t-1}, T_t$ carry through (treating the missing side as $0$). The sum is clipped per cluster to $[-0.95, +0.95]$ so repeated reinforcement cannot drive the weight unboundedly. Use this when prompts are *layered refinements* of an evolving query.

| Cluster | $T_{t-1}$ | $T_t$ | $T_t^{\mathrm{eff}}$ |
| --- | --- | --- | --- |
| $c_{17}$ | $+0.6$ | — | $+0.6$ |
| $c_{42}$ | $-0.4$ | $+0.5$ | $+0.1$ |
| $c_{99}$ | — | $+0.3$ | $+0.3$ |

The sci-fi boost is kept, romance partially flips from $-0.4$ to $+0.1$, and a new cluster $c_{99}$ enters at $+0.3$.

#### 4.5.3 `intersect`

$$
T_t^{\mathrm{eff}}(c) \;=\; T_t(c), \qquad c \in T_{t-1} \cap T_t.
$$

The intersection of the two prompts' cluster sets, valued by the **current** iteration's weights. Clusters that appear in only one prompt are dropped. Useful for diagnostic studies that test how much expressiveness survives a narrowing pipeline.

| Cluster | $T_{t-1}$ | $T_t$ | $T_t^{\mathrm{eff}}$ |
| --- | --- | --- | --- |
| $c_{17}$ | $+0.6$ | — | — |
| $c_{42}$ | $-0.4$ | $+0.5$ | $+0.5$ |
| $c_{99}$ | — | $+0.3$ | — |

Only the cluster both prompts agree on survives; new clusters and old clusters are both pruned.

#### 4.5.4 Why `replace` and `add` look similar in casual use

When two consecutive prompts target the same single cluster with the same sign, `replace` and `add` produce **almost identical** $T^{\mathrm{eff}}$ — `replace` writes $T_t(c)$, `add` writes $T_{t-1}(c) + T_t(c)$, but both drive the same single direction so the qualitative recommendation behaviour looks similar.

The differences become visible when:

- prompts target **different clusters** — `add` keeps the old one, `replace` discards it;
- prompts have **opposite signs** for the same cluster — `add` (partially) cancels them out, `replace` jumps cleanly to the new direction;
- prompts are **repeated reinforcements** — `add` saturates at $\pm 0.95$ after enough iterations, `replace` always stays at the latest single-iteration magnitude.

Reference implementation (one function, ten lines):

```44:56:server/plugins/steering/routes/steering/actions.py
def _compose_text_adjustments(mode: str, previous: dict, current: dict) -> dict:
    mode = (mode or DEFAULT_TEXT_COMPOSITION_MODE).strip().lower()
    if mode not in SUPPORTED_TEXT_COMPOSITION_MODES:
        mode = DEFAULT_TEXT_COMPOSITION_MODE
    if not previous or mode == "replace":
        return dict(current or {})
    if mode == "intersect":
        keys = set(previous.keys()) & set((current or {}).keys())
        return {key: float(current[key]) for key in keys}
    merged = {key: float(value) for key, value in (previous or {}).items()}
    for key, value in (current or {}).items():
        merged[key] = round(max(-0.95, min(0.95, float(merged.get(key, 0.0)) + float(value))), 2)
    return merged
```

#### NFR-12 no-match

If $\mathrm{total}(c) \le 0$ for **all** clusters, the route returns `status = no-match` with HTTP 200, a `SaeTextSteeringQuery` row is still written (empty matches), and the UI shows *"We could not match your text to any feature, try different wording."*

#### Scope

The "previous prompt" $T_{t-1}$ is namespaced by `(study_guid, phase_index)`. A prompt issued in approach A's phase $0$ cannot accidentally compose with a prompt issued later in approach B's phase $1$, even though both share a Flask session. See [`design-decisions.md` Section 21](design-decisions.md#21-text-steering-state-is-namespaced-by-study_guid-phase_index-and-the-ui-surface-is-reset-per-iteration) for the leakage fix.

## 5. Example-based steering (FR-08)

#### Input

A set of example movie ids $E = \{e_1, \dots, e_n\}$ (typically the participant's current likes, plus any explicitly picked example movies).

### 5.1 Mean activation

For the active SAE, look each $e_j$ up in `recommender.item_features` and take the element-wise mean across the matched rows (movies not present in the SAE matrix are silently skipped):

$$
\mu \;=\; \frac{1}{|E^\ast|} \sum_{e \in E^\ast} \mathrm{sae\_activation}(e),
$$

where $E^\ast \subseteq E$ is the matched subset and $\mathrm{sae\_activation}(e) \in \mathbb{R}^n$.

### 5.2 Per-cluster score and weight

For each cluster $c$ with neuron set $M[c]$, define

$$
\mathrm{score}_e(c) \;=\; \frac{1}{|M[c]|} \sum_{n \in M[c]} \mu_n.
$$

Only clusters with $\mathrm{score}_e(c) > 0$ survive. The weight is bounded with strength $s = $ `example_selection_weight` (default `0.65`) and a sub-linear score boost (factor `0.6`, upper bound `0.95` — both hardcoded):

$$
w_e(c) \;=\; \mathrm{clip}\!\bigl(\, s \cdot (1 + 0.6 \cdot \mathrm{score}_e(c)),\; 0,\; 0.95 \,\bigr).
$$

The top `example_selection_top_k` clusters (default `6`) by $\mathrm{score}_e(c)$ are written to `sae_example_steering` + children; the others are dropped.

```62:87:server/plugins/steering/modalities/examples.py
    mean_activation = np.mean(np.asarray(activations), axis=0)
    cluster_rows: List[Dict] = []
    for cluster in semantic_clusters.get("clusters", []):
        neuron_ids = cluster.get("neuron_ids", [])
        values = [float(mean_activation[nid]) for nid in neuron_ids if nid < len(mean_activation)]
        if not values:
            continue
        cluster_score = float(np.mean(values))
        if cluster_score <= 0:
            continue
        strength = min(1.0, max(0.0, float(default_weight)))
        weight = min(0.95, max(0.0, strength * (1.0 + (cluster_score * 0.6))))
        cluster_rows.append(
            {
                "id": cluster.get("cluster_id"),
                "label": cluster.get("label") or str(cluster.get("cluster_id")),
                "description": cluster.get("description", ""),
                "weight": round(weight, 2),
                "direction": "boost",
                "activation_score": round(cluster_score, 4),
                "member_ids": neuron_ids,
            }
        )

    cluster_rows.sort(key=lambda row: (-row["activation_score"], row["label"].lower()))
    cluster_rows = cluster_rows[:top_k]
```

#### Direction

Example-based steering produces only positive weights (`direction = boost`); suppression has no analogue here because the UI has no "anti-example" concept.

#### Composition

Whether example weights are merged with the slider/toggle cumulative map is governed by the per-approach flag `use_selected_movies_as_examples` (default `false`).

- When **on**, the iteration controller runs `ExampleSteering` on the participant's current likes every iteration and merges its output with the cumulative slider/toggle map via `_merge_adjustments` (additive, then values with $|\cdot| < 10^{-3}$ are dropped).
- When **off**, likes drive only the ELSA seed re-weighting ([Section 7](#7-elsa-seed-re-weighting-from-likes)); the example modality is invoked only by an explicit `POST /apply-example-steering` call.

In either case, example adjustments do not stack across iterations on their own — each call to `/apply-example-steering` overwrites `session["last_example_steering"]`.

```181:195:server/plugins/steering/service/iteration_controller.py
    example_adjustments = {}
    example_metadata = {}
    if active_model_cfg.get("use_selected_movies_as_examples") and current_liked_set:
        example_result = get_modality_strategy(Modalities.EXAMPLES).apply(
            {"example_movie_ids": list(current_liked_set)},
            conf=conf,
            active_model=active_model_cfg,
        )
        example_adjustments = expand_feature_adjustments(
            raw_adjustments=example_result.adjustments,
            cluster_map=cluster_map,
        )
        example_metadata = example_result.metadata or {}

    recommendation_adjustments = _merge_adjustments(feature_adjustments, example_adjustments)
```

## 6. Reset (FR-12)

Reset is a degenerate modality: it has no weights of its own. `POST /reset` clears the in-session steering state **and** the in-session liked-movie state, then writes one `SaeResetAction` row plus a `SaeSteeringEvent` envelope so the audit timeline records when the participant chose to start over.

The preference-elicitation pool $E_0$ (the movies the participant picked before steering started) is **never** touched — only the post-elicitation memory. Three simultaneous identities at iteration $t+1$:

*Neuron offsets wiped.*

$$
\Delta^{(t+1)}_n \;=\; 0 \quad \forall n.
$$

*Liked set wiped.*

$$
L^{(t+1)} \;=\; \varnothing.
$$

*ELSA seed reverts to the elicitation mean.*

$$
\hat{s}^{(t+1)} \;=\; \frac{1}{|E_0|} \sum_{m \in E_0} \mathrm{emb}(m).
$$

The session keys explicitly emptied are: `cumulative_adjustments`, `feature_adjustments`, `user_touched_features`, `excluded_movies_from_text`, `last_text_steering`, `last_example_steering`, `boosted_liked_ids`, and the current phase's entry in `persistent_liked_by_phase`. `update_elsa_seed_with_likes(set(), …)` is then called so the ELSA seed ([Section 7](#7-elsa-seed-re-weighting-from-likes)) reverts to the elicitation-only mean.

The frontend mirrors this: `resetAllControls()` zeroes every slider's `featureAdjustments`, drops the detected-tags container, empties `likedMovies` and `likedInIteration`, and re-syncs every previously-highlighted recommendation card so the heart selection visually disappears.

The single `SaeResetAction` row in the DB is the source of truth that the reset happened — individual per-movie unlikes are deliberately not re-logged (see [`design-decisions.md` Section 5](design-decisions.md#5-reset-is-a-dedicated-endpoint-not-a-flag-on-adjust-features)).

## 7. ELSA seed re-weighting from likes

The ELSA seed embedding $\hat{s} \in \mathbb{R}^d$ is a weighted mean of two pools: the original elicitation movies $E_0$ (weight `1` each, hardcoded) and the participant's current liked movies $L$, capped at the first $K$ ids (sorted ascending) and weighted by $\lambda$ per liked movie:

$$
\hat{s} \;=\; \frac{\sum_{m \in E_0} \mathrm{emb}(m) \;+\; \lambda \cdot \sum_{m \in L^{\le K}} \mathrm{emb}(m)}{|E_0| \;+\; \lambda \cdot |L^{\le K}|},
$$

where $L^{\le K} = \mathrm{sorted}(L)[\,:K]$ and the cardinalities count only ids that resolve in `recommender.item_ids`.

| Symbol | Meaning | Source |
| --- | --- | --- |
| $E_0$ | Preference-elicitation movies | `session["elicitation_selected_movies"]` |
| $L$ | Current liked movies in this approach | from the request body each iteration |
| $\lambda$ | Per-like weight | `selection_signal_weight` (default `0.5`, or `0.25` if the approach uses sliders/toggles/hybrid) |
| $K$ | `like_cap`, max likes counted | `10`, hardcoded at the call site in `iteration_controller.py` |

The seed is **recomputed from the elicitation pool every iteration** — it is not a running blend $(1-\lambda) \hat{s}_\mathrm{old} + \lambda \cdot \dots$ The seed is then consumed by the recommender as the query vector for the CF term (see [Section 1.2](#12-how-the-expanded-vector-enters-ranking) and [Section 10](#10-reranking-strategies-reranking_strategy-config-key)).

When `interaction_mode` is `reset` ([Section 2.1](#21-interaction-history-mode-cumulative-vs-reset)), the iteration controller calls `update_elsa_seed_with_likes(set(), …)` so $L$ is forced empty and $\hat{s}$ collapses to the pure elicitation mean — no like signal carries into the next iteration. In `cumulative` mode, the controller passes the participant's actual liked set, giving the formula above.

```27:54:server/plugins/steering/service/engine.py
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
```

## 8. Final ranking — cross-strategy summary

All three reranking strategies use the same three per-item building blocks:

$$
\mathrm{cf}(i) \;=\; \cos(e_i,\, \hat{s}) \cdot w_{\mathrm{cf}}, \qquad
\mathrm{genre}(i) \;=\; j_i \cdot w_g, \qquad
\mathrm{sae}(i) \;=\; f_i^\top a.
$$

What changes is **how** the SAE signal enters the final score:

| Strategy | Final score $\mathrm{score}(i)$ | Where SAE enters | Has $\gamma$ / clamp? |
| --- | --- | --- | --- |
| `feature-conditioned` | $\mathrm{cf}(i) + \mathrm{genre}(i) + \mathrm{clip}(\gamma \cdot \mathrm{sae}(i), -c, +c) + w_{\mathrm{prior}} \cdot \tilde{\mathrm{base}}(i)$ | additive score term | yes (adaptive) |
| `latent-perturbation` | $\cos(e_i,\, \hat{s}') \cdot w_{\mathrm{cf}} + \mathrm{genre}(i)$, with $\hat{s}'$ from the rotated seed | seed rotation, then CF | no (fixed $\alpha$) |
| `constrained-subset` | $\mathrm{cf}(i) + \mathrm{genre}(i)$ on items where $\mathrm{sae}(i) \ge \tau^\ast$; $-\infty$ otherwise (fallback to base if mask is empty) | hard filter, then CF | no (fixed $\tau$) |

The full math for each strategy is in [Section 10](#10-reranking-strategies-reranking_strategy-config-key); the rationale for the *set* of strategies is in [`design-decisions.md` Section 23](design-decisions.md#23-reranking-strategies-reranking_strategy-config-key).

## 9. Notation summary

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $e_i$ | $\mathbb{R}^d$ | Row $i$ of `recommender.item_embeddings` — the ELSA dense embedding for item $i$. $d$ is the CF embedding dim (typically $256$). |
| $\hat{s}$ | $\mathbb{R}^d$ | The user seed embedding (`session["elsa_seed"]`, see [Section 7](#7-elsa-seed-re-weighting-from-likes)), $L_2$-normalised at use time. |
| $f_i$ | $\mathbb{R}^n$ | Row $i$ of `recommender.item_features` — the SAE feature activations for item $i$. $n$ is the SAE feature count ($1024$ for `TopKSAE-1024`). |
| $a$ | $\mathbb{R}^n$ | The per-neuron *sae_profile* built from `feature_adjustments` after cluster-to-neuron expansion ([Section 1.1](#11-cluster-to-neuron-expansion)). Sparse. |
| $W_{\mathrm{dec}}$ | $\mathbb{R}^{n \times d}$ | SAE decoder weight (`sae_model.decoder_w`), mapping feature space back to embedding space. |
| $j_i$ | $\mathbb{R}_{\ge 0}$ | Genre Jaccard bonus for item $i$ (precomputed by the caller). |
| $w_{\mathrm{cf}}, w_g$ | scalars | Blend weights from `build_blend_plan(feature_adjustments)`. Two regimes: `profile_prior` (no explicit steering) and `steering_primary` (any non-zero $a$). |
| $M[c]$ | finite set of $n$ | Neuron set of cluster $c$. |

## 10. Reranking strategies (`reranking_strategy` config key)

The study-level config key `reranking_strategy` controls how the three building blocks (CF, genre, SAE) are combined into the final per-item score. Three legal values are enumerated in `constants.py`:

```52:57:server/plugins/steering/constants.py
DEFAULT_RERANKING_STRATEGY = "feature-conditioned"
SUPPORTED_RERANKING_STRATEGIES = {
    "feature-conditioned",
    "latent-perturbation",
    "constrained-subset",
}
```

The strategy is captured at approach-run creation (`SaeApproachRun.reranking_strategy`) and threaded through the recommender call:

```117:127:server/plugins/steering/recommendation/service.py
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
```

All three strategies share the per-item building blocks defined in [Section 8](#8-final-ranking--cross-strategy-summary); they differ in *where* the SAE signal enters.

### 10.1 `feature-conditioned` (default — additive blend)

This is the production default; it is what every existing pilot has used and what the dashboard analytics have been validated against. The SAE signal is added to the base $\mathrm{cf} + \mathrm{genre}$ score with **adaptive** gain $\gamma$ and per-iteration clamp $c$:

$$
\mathrm{score}(i) \;=\;
\underbrace{\mathrm{cf}(i) + \mathrm{genre}(i)}_{\mathrm{base}(i)}
\;+\; \underbrace{\mathrm{clip}\!\bigl(\gamma \cdot \mathrm{sae}(i),\; -c,\; +c\bigr)}_{\text{steering}(i)}
\;+\; \underbrace{w_{\mathrm{prior}} \cdot \tilde{\mathrm{base}}(i)}_{\text{tiebreak}},
$$

where $\tilde{\mathrm{base}}(i) \in [0, 1]$ is the min–max normalised $\mathrm{base}$ score over allowed items (zero when the span is too small).

Three regimes for $(\gamma, c)$ — computed per iteration over the allowed candidate set $\mathcal{A}$:

| Regime | When | $\gamma$ | $c$ |
| --- | --- | --- | --- |
| **No adjustments** | $\lVert a \rVert = 0$ | $0$ | $0$ |
| **Steering primary** | blend plan is `steering_primary` (any $\lvert a_n \rvert > 10^{-6}$) | $1$ | $\max_{i \in \mathcal{A}} \lvert \mathrm{sae}(i) \rvert$ |
| **Moderate, big pool** | otherwise and $\lvert \mathcal{A} \rvert \ge 10$ | $\mathrm{clip}\!\left(0.30 \cdot \frac{\mathrm{IQR}(\mathrm{base})}{\mathrm{IQR}(\mathrm{sae})},\; 0.03,\; 0.35\right)$ | $\max\!\bigl(0.35 \cdot \mathrm{span}(\mathrm{base}),\; 0.05 \cdot \mathrm{span}(\mathrm{sae})\bigr)$ |
| **Moderate, small pool** | otherwise and $\lvert \mathcal{A} \rvert < 10$ | $0.15$ | $2.0$ |

Here $\mathrm{IQR}$ is the inter-quartile range and $\mathrm{span}$ is the $p_{05} \to p_{95}$ range, both taken over $\mathcal{A}$. The adaptive $\gamma$ scales the SAE term to match the dispersion of the base score, so neither signal dominates by accident.

Source: `recommendation/sae_recommender.py`, the `feature-conditioned` branch in `get_recommendations`.

### 10.2 `latent-perturbation` (rotate the seed; pure CF rank)

Instead of *adding* an SAE-score term after the CF term, this strategy **moves the user-seed embedding** by an SAE-derived direction and then ranks with pure CF on the moved seed. Conceptually: *the user's steering is a refinement of who they are, not a post-hoc bump of what they see.*

**Direction decoding** (SAE decoder back to embedding space):

$$
d \;=\; W_{\mathrm{dec}}^{\top} a \;\in\; \mathbb{R}^d, \qquad
\hat{d} \;=\; \frac{d}{\lVert d \rVert}.
$$

**Perturbed seed** (with $\hat{s}$ already $L_2$-normalised):

$$
\hat{s}' \;=\; \frac{\hat{s} + \alpha \cdot \hat{d}}{\bigl\lVert \hat{s} + \alpha \cdot \hat{d} \bigr\rVert}.
$$

**Final score:**

$$
\mathrm{score}(i) \;=\; \cos(e_i,\; \hat{s}') \cdot w_{\mathrm{cf}} \;+\; \mathrm{genre}(i).
$$

**No additive steering term.** The debug payload reports `steering_score = 0` for every item; the observable influence shows up as a change in `cf_score`.

**Parameter.** $\alpha \in [0, 1]$ (`latent_perturbation_alpha`, default `0.30`). Larger $\alpha$ means a more aggressive rotation. The value is intentionally capped well below `1.0` because the seed is normalised — pushing $\alpha \to 1$ moves the seed almost entirely onto the decoded direction, which often falls outside the CF-trained manifold and degenerates into noise.

Source: `recommendation/sae_recommender.py`, the `latent-perturbation` branch and the `_decode_sae_profile_to_embedding_space` helper.

### 10.3 `constrained-subset` (hard $\tau$ filter, CF rank inside)

This strategy enforces a **hard membership constraint**: an item only enters the recommendation list if its SAE score is at least a fraction $\tau$ of the strongest positive SAE score in the allowed candidate set. Within the surviving subset, ranking is by base $\mathrm{cf} + \mathrm{genre}$ (no additive SAE term).

Let $\mathcal{A}$ be the allowed candidate set, $S = \{\mathrm{sae}(i) : i \in \mathcal{A}\}$, and $S^+ = \{ s \in S : s > 0 \}$. Define the threshold

$$
\tau^\ast \;=\;
\begin{cases}
\tau \cdot \max(S^+) & \text{if } S^+ \ne \varnothing, \\
0 & \text{otherwise,}
\end{cases}
$$

and the per-item mask

$$
m_i \;=\; \mathbb{1}\bigl[\,\mathrm{sae}(i) \ge \tau^\ast\,\bigr].
$$

**Final score:**

$$
\mathrm{score}(i) \;=\;
\begin{cases}
\mathrm{cf}(i) + \mathrm{genre}(i) & m_i = 1, \\
-\infty & \text{otherwise.}
\end{cases}
$$

**Fallback.** If $\sum_i m_i = 0$ (e.g. the user has not adjusted anything, so every SAE score is $0$, or every item is genuinely below the bar), the filter is silently dropped and ranking falls back to pure base CF + genre. The recommender never returns an empty list because of this strategy.

**Parameter.** $\tau \in [0, 1]$ (`constrained_subset_tau`, default `0.25`). Larger $\tau$ means a stricter filter — too high and the fallback kicks in often; too low and the strategy degenerates into "no filter, just CF rank".

Source: `recommendation/sae_recommender.py`, the `constrained-subset` branch.

### 10.4 Why all three strategies exist

| Property | `feature-conditioned` | `latent-perturbation` | `constrained-subset` |
| --- | --- | --- | --- |
| SAE enters as | additive score term | seed rotation, then CF | hard filter, then CF |
| Has adaptive gain / clamp? | yes (adaptive $\gamma, c$) | no (single $\alpha$) | no (single $\tau$) |
| Top-1 can be "off-target"? | yes — steering can fail to clear $\mathrm{base} + \mathrm{clamp}$ | yes — rotation is gentle | no when the filter has survivors (every returned item satisfies the SAE threshold by construction); on fallback the strategy is honest about it via `debug.constrained_subset_survivors` |
| Falls back when SAE signal is empty? | yes — steering term becomes $0$ | yes — no perturbation applied | yes — mask is dropped if no positive SAE score |
| Suitable as | baseline | ablation: "is the SAE signal informative even without explicit boosting?" | upper bound of *guaranteed* steering, ignoring CF gradients |

The production system stays on `feature-conditioned` because that is what has been piloted. The other two are available as research toggles and are persisted on every `SaeApproachRun` row so retrospective analysis can match strategy to outcome.

---

## Appendix: What is configurable

Every numeric constant in the formulas above is either a **config key** (researcher can change it per study or per approach) or **hardcoded** (intentionally fixed in code). This table is the single source of truth; inline mentions only use the symbol.

| Symbol in formulas | Code name | Default | Scope | Read from |
| --- | --- | --- | --- | --- |
| $\alpha$ (slider amplification) | `SLIDER_AMPLIFICATION` | `2.0` | **hardcoded** | `modalities/sliders.py` |
| $\beta$ (toggle magnitude) | `toggle_default_weight` | `0.65` | per-approach (study-level fallback) | `active_model` → `conf` |
| $w^\ast$ (text baseline) | `text_steering_weight` | `0.55` | per-approach (study-level fallback) | `active_model` → `conf` |
| `top-K` (text) | `text_steering_top_k` | `6` | per-approach (study-level fallback) | `active_model` → `conf` |
| text composition | `text_composition_mode` ∈ {`replace`, `add`, `intersect`} | `replace` | per-approach (study-level fallback `text_steering.composition_mode`) | `active_model` → `conf["text_steering"]` |
| $\mathrm{max\_chars}$ | `text_steering.max_query_chars` | `200` | study-level | `conf["text_steering"]` |
| $s$ (example strength) | `example_selection_weight` | `0.65` | per-approach (study-level fallback) | `active_model` → `conf` |
| `top-K` (examples) | `example_selection_top_k` | `6` | per-approach (study-level fallback) | `active_model` → `conf` |
| Examples merged with sliders? | `use_selected_movies_as_examples` | `false` | per-approach (study-level fallback) | `active_model` → `conf` |
| $\lambda$ (like weight for seed) | `selection_signal_weight` | `0.5` (or `0.25` with sliders/toggles/hybrid) | per-approach (study-level fallback) | `active_model` → `conf` |
| $K$ (`like_cap`) | — | `10` | **hardcoded** at the call site | `iteration_controller.py` |
| SAE checkpoint | `sae` | `DEFAULT_TOPK_SAE_MODEL_ID` | per-approach | `active_model` |
| Reranking strategy | `reranking_strategy` ∈ {`feature-conditioned`, `latent-perturbation`, `constrained-subset`} | `feature-conditioned` | study-level | `conf` |
| $\alpha$ (latent perturbation) | `latent_perturbation_alpha` | `0.30` | study-level (or per-approach via `models[*].latent_perturbation_alpha`) | `conf` |
| $\tau$ (constrained subset) | `constrained_subset_tau` | `0.25` | study-level (or per-approach via `models[*].constrained_subset_tau`) | `conf` |
| $\gamma, c$ (`feature-conditioned`) | adaptive | computed per iteration from candidate-set quantiles | **hardcoded** | `recommendation/sae_recommender.py` |
| Token-match weights `2.5 / 1.25 / 0.75`, phrase bonus `3.0`, coverage term | — | as shown | **hardcoded** | `modalities/text.py` |
| Intensity ladder $\{0.65, 1.0, 1.35\}$, `_NEGATIVE_HINTS`, `_INTENSITY_*`, `_STOP_WORDS` | — | as shown | **hardcoded** | `modalities/text.py` |
| Weight bounds $[0.25, 0.95]$, $[0.1, 0.95]$ (text); $[0, 0.95]$ (examples); $[-0.95, 0.95]$ (text `add` mode) | — | as shown | **hardcoded** | `modalities/text.py`, `modalities/examples.py`, `routes/steering/actions.py` |
| Example score boost factor `0.6` | — | `0.6` | **hardcoded** | `modalities/examples.py` |
| Drop thresholds $10^{-3}$ (slider/toggle) and $10^{-4}$ (cluster expansion) | — | as shown | **hardcoded** | `modalities/*.py`, `recommendation/semantic_registry.py` |

*"Per-approach (study-level fallback)"* means that each approach in `conf["models"]` carries its own value; if it is unset on the approach, `normalize_study_config` copies the study-level value down before the iteration loop reads it. Researchers do not need to set both — the create-study form fills the approach values from the study-level defaults.
