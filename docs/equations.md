# Equations and Scoring

This page is the math reference for the SAE Steering plugin. Each modality has its own section with the same shape: **input → per-cluster weight → composition → contract surface**. Every formula is paired with a short code snippet so the math and the code can be checked against each other directly.

Cross-references:

- [`tech-docs.md`](tech-docs.md) §6 — the iteration loop that consumes these weights.
- [`design-decisions.md`](design-decisions.md) §6, §7, §8 — rationale for composition modes, the no-match fallback, and the reranking enum.
- [`formative-examples.md`](formative-examples.md) §3, §5 — how to add a new modality / a new reranking strategy.

## 1. Common framework

All four user-facing modalities in `server/plugins/steering/modalities/` implement the same strategy interface:

```16:22:server/plugins/steering/modalities/base.py
class SteeringModality:
    """Small strategy interface for one user-facing steering modality."""

    modality_id: str = ""

    def apply(self, data: Dict[str, Any], *, conf: dict, active_model: dict) -> SteeringResult:
        raise NotImplementedError
```

A `SteeringResult` is `(features, adjustments, metadata)` where `adjustments : cluster_id → weight w(c) ∈ [-1, 1]`. The semantics is "boost items that activate cluster `c`'s neurons" if `w(c) > 0`, "suppress" if `w(c) < 0`.

### 1.1 Cluster → neuron expansion

The recommender ranks items by an inner product against a per-**neuron** profile, not a per-cluster one. The bridge is `expand_feature_adjustments`: each cluster delta `δ_c` is fanned out to every neuron `n ∈ cluster_map[c]`, and overlapping clusters sum:

$$
\Delta_n \;=\; \sum_{c\,:\,n \in c} \delta_c
$$

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

Each item `i` has a row `f_i` of SAE activations of length `n_features`. The recommender builds a sparse profile `a` with `a_n = Δ_n` (after expansion) and computes the SAE term as the inner product `f_i · a`, implemented as a matrix–vector product:

```381:396:server/plugins/steering/recommendation/sae_recommender.py
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
```

The full final score `cf + genre + clamp(γ · sae) + tiebreak` is summarized in §8; the per-modality math below covers only how each modality produces its slice of `δ_c`.

## 2. Sliders (FR-05/06)

**Input.** UI sends a dict `raw_adjustments[cluster_id] = δ_c ∈ [-1, 1]`, one slider per visible cluster.

**Per-iteration contribution.** `SliderSteering` multiplies each non-trivial delta by a fixed amplification factor `α = SLIDER_AMPLIFICATION = 2.0` (**hardcoded** in `modalities/sliders.py`; intentionally not exposed in the study config so that two studies remain directly comparable) so user-visible "small" moves still produce a non-trivial score gap:

$$
w_{\text{slider}}(c) \;=\; \alpha \cdot \delta_c \quad\text{for }|\delta_c| > 10^{-3}
$$

```17:28:server/plugins/steering/modalities/sliders.py
class SliderSteering(SteeringModality):
    
    modality_id = Modalities.SLIDERS

    def apply(self, data: Dict[str, Any], *, conf: dict, active_model: dict) -> SteeringResult:
        raw_adjustments = data.get("adjustments", {}) or {}
        adjustments = {
            str(feature_id): round(float(value) * SLIDER_AMPLIFICATION, 4)
            for feature_id, value in raw_adjustments.items()
            if abs(float(value)) > 0.001
        }
        return SteeringResult(features=[], adjustments=adjustments, metadata={"raw_adjustments": raw_adjustments})
```

**Composition across iterations.** Sliders **accumulate**, not replace. The iteration controller keeps `previous_adjustments : neuron_id → weight` in the session and, after expanding the cluster deltas to neurons, adds the amplified increment per neuron:

$$
\Delta_n^{(t)} \;=\; \Delta_n^{(t-1)} \,+\, \alpha \cdot \delta_n^{(t)}
$$

```109:117:server/plugins/steering/service/iteration_controller.py
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

A subsequent slider refresh (`compute_updated_sliders`) hides clusters that have already been touched in this phase, so the participant always sees fresh content to steer with — touched clusters are not erased, only filtered out of the next pool. The cumulative map stays in `session["cumulative_adjustments"]`.

## 3. Toggles (FR-07)

**Input.** UI sends `raw_adjustments[cluster_id] ∈ ℝ`, where only the **sign** is meaningful: positive → boost, negative → suppress, near-zero → unset.

**Per-iteration contribution.** Each touched cluster gets a **fixed-magnitude** weight `β = toggle_default_weight` (config key, per-approach with study-level fallback; default `0.65`), signed by the user's choice:

$$
w_{\text{toggle}}(c) \;=\; \operatorname{sign}(\delta_c) \cdot \beta \quad\text{for }|\delta_c| > 10^{-3}
$$

```14:26:server/plugins/steering/modalities/toggles.py
class ToggleSteering(SteeringModality):
    modality_id = Modalities.TOGGLES

    def apply(self, data: Dict[str, Any], *, conf: dict, active_model: dict) -> SteeringResult:
        raw_adjustments = data.get("adjustments", {}) or {}
        default_weight = float(active_model.get("toggle_default_weight", conf.get("toggle_default_weight", DEFAULT_TOGGLE_WEIGHT)))
        adjustments = {}
        for feature_id, value in raw_adjustments.items():
            numeric = float(value)
            if abs(numeric) <= 0.001:
                continue
            sign = 1.0 if numeric > 0 else -1.0
            adjustments[str(feature_id)] = round(sign * default_weight, 4)
```

**Composition across iterations.** Same accumulation path as sliders (the iteration controller treats the post-expansion neuron deltas uniformly; `SLIDER_AMPLIFICATION = 2.0` is applied again at the neuron step). The practical consequence is that toggling boost twice for the same cluster reinforces it, and toggling boost then suppress mostly cancels out.

## 4. Text steering (FR-09)

**Input.** Free-form participant query `Q` (≤ `text_steering.max_query_chars`, study-level config; default `200`; the route returns 400 if longer). Length is enforced at the entry of `/parse-text-steering`.

### 4.1 Segmentation

`Q` is split on the regex `_SEGMENT_BOUNDARY_RE = [.;]|\bbut\b|\bhowever\b`. Each non-empty chunk becomes a segment `s_i` with three derived quantities:

- `direction(s_i) ∈ {+1, -1}` — `-1` iff any marker in `_NEGATIVE_HINTS` (`not`, `no`, `never`, `don't`, `i hate`, …) appears in `s_i`, else `+1`. *(hint list and segmentation regex are hardcoded.)*
- `intensity(s_i) ∈ {0.65, 1.0, 1.35}` — `1.35` for "much/way/a lot more|less" or "strongly/definitely"; `0.65` for "slightly/a bit/somewhat/kind of"; `1.0` otherwise. *(ladder and triggers are hardcoded.)*
- `tokens(s_i)` — alphanumeric tokens of length ≥ 2 with `_STOP_WORDS` removed. *(stop-list is hardcoded.)*

```103:119:server/plugins/steering/modalities/text.py
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

For each cluster `c` with label `l_c` and description `d_c`, and each segment `s_i`:

$$
\text{score}(s_i, c) \;=\; 3 \cdot \mathbb{1}\big[\text{text}(s_i) \subseteq l_c \cup d_c\big]
\;+\; \sum_{t \in tokens(s_i)} w(t, c)
\;+\; \frac{|tokens(s_i) \cap \text{tokens}(l_c \cup d_c)|}{|tokens(s_i)|}
$$

with

$$
w(t, c) =
\begin{cases}
2.5 & t \in tokens(l_c)\\
1.25 & t \in tokens(d_c)\\
0.75 & t \in (l_c \cup d_c)\text{ but no token match}\\
0 & \text{otherwise}
\end{cases}
$$

```133:157:server/plugins/steering/modalities/text.py
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

For each cluster `c`, contributions are summed across segments that score positive, with `S_c = \{ i : \text{score}(s_i, c) > 0 \}`:

$$
\text{total}(c) = \sum_{i \in S_c} \text{score}(s_i, c)
$$

$$
\overline{\text{dir}}(c) =
\begin{cases}
+1 & \sum_{i \in S_c} direction(s_i) \geq 0\\
-1 & \text{otherwise}
\end{cases}
\qquad
\overline{\text{int}}(c) = \frac{1}{|S_c|} \sum_{i \in S_c} intensity(s_i)
$$

### 4.4 Weight assignment

With `w* = text_steering_weight` (config key, per-approach with study-level fallback; default `0.55`). The bounds `[0.25, 0.95]` and `[0.1, 0.95]` are hardcoded:

$$
w_0(c) = \min\!\Big(0.95,\; \max\!\big(0.25,\; w^\ast + \tfrac{\text{total}(c)}{10}\big)\Big)
$$

$$
w(c) = \overline{\text{dir}}(c) \cdot \min\!\Big(0.95,\; \max\big(0.1,\; w_0(c) \cdot \overline{\text{int}}(c)\big)\Big)
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

### 4.5 Top-K and composition across iterations

The top `text_steering_top_k` clusters (config key, per-approach with study-level fallback; default `6`) by `|w(c)|` survive; the others are dropped. Their `(cluster_id, w(c))` map is the iteration's text adjustments.

Composition with the previous iteration's map is governed by the active approach's `text_composition_mode` (config key; `text_steering.composition_mode` is the study-level fallback). The three supported values are:

- **`replace`** (default) — iteration `t` overwrites iteration `t-1`.
- **`add`** — sum per cluster, clipped to `[-0.95, 0.95]` *(clip is hardcoded)*.
- **`intersect`** — keep only clusters present in both iterations; use iteration `t`'s weight.

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

**NFR-12 no-match.** If `total(c) ≤ 0` for **all** clusters, the route returns `status = no-match` with HTTP 200, a `SaeTextSteeringQuery` row is still written (empty matches), and the UI shows "We could not match your text to any feature, try different wording."

## 5. Example-based steering (FR-08)

**Input.** A set of example movie ids `E = \{e_1, …, e_n\}` (typically the participant's current likes plus any explicitly picked example movies).

### 5.1 Mean activation

For the active SAE, look each `e_j` up in `recommender.item_features` and take the element-wise mean across the matched rows (movies that are not present in the SAE matrix are silently skipped):

$$
\mu \;=\; \frac{1}{|E^\ast|} \sum_{e \in E^\ast} \text{sae\_activation}(e)
$$

where `E* ⊆ E` is the matched subset.

### 5.2 Per-cluster score and weight

For each cluster `c` with neuron set `c = \{n_1, …, n_k\}`:

$$
\text{score}_e(c) \;=\; \frac{1}{|c|} \sum_{n \in c} \mu_n
$$

Only clusters with `score_e(c) > 0` survive. The weight is bounded with strength `s = example_selection_weight` (config key, per-approach with study-level fallback; default `0.65`) and a sub-linear score boost; the boost factor `0.6` and the upper bound `0.95` are hardcoded:

$$
w_e(c) \;=\; \min\!\Big(0.95,\; \max\!\big(0,\; s \cdot (1 + 0.6 \cdot \text{score}_e(c))\big)\Big)
$$

The top `example_selection_top_k` clusters (config key, per-approach with study-level fallback; default `6`) by `score_e(c)` are written to `sae_example_steering` + children; the others are dropped.

```48:73:server/plugins/steering/modalities/examples.py
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

**Direction.** Example-based steering only produces positive weights (`direction = boost`); suppression has no analogue here because there is no "anti-example" concept in the UI.

**Composition (configurable).** Whether example weights are merged with the slider/toggle cumulative map at all is governed by the per-approach flag `use_selected_movies_as_examples` (config key, per-approach with study-level fallback; default `false`). When the flag is on, the iteration controller runs the `ExampleSteering` strategy on the participant's current likes every iteration and merges its output with the cumulative slider/toggle map via `_merge_adjustments` (additive, then values with `|·| < 10⁻³` are dropped). When the flag is off, likes drive only the ELSA seed re-weighting (§7) and the example modality is invoked only by the explicit `/apply-example-steering` route. In either case, example adjustments do not stack across iterations on their own — each call to `/apply-example-steering` overwrites `session["last_example_steering"]`.

```129:143:server/plugins/steering/service/iteration_controller.py
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

Reset is a degenerate modality: it has no weights of its own. `POST /reset` clears the in-session cumulative state and writes one `SaeResetAction` row plus a `SaeSteeringEvent` envelope so the timeline records when the participant chose to start over:

$$
\Delta^{(t+1)}_n \;=\; 0 \quad \forall n,\qquad \text{session cleared}
$$

```288:316:server/plugins/steering/routes/steering/actions.py
@bp.route("/reset", methods=["POST"])
def reset_steering():
    """Dedicated reset endpoint (FR-12).

    Clears the in-session adjustment state and writes one SaeResetAction row +
    one SaeSteeringEvent envelope. Returns the cleared state so the UI can mirror
    it without round-tripping through /adjust-features.
    """
    try:
        data = request.get_json(force=True) or {}
        scope = str(data.get("scope") or "all-features")
        trigger = str(data.get("trigger") or "manual-ui-reset")
        conf = normalize_study_config(load_user_study_config(session.get("user_study_id")))
        active_model = get_active_model_config(conf)
        participation_id = session.get("participation_id")
        if participation_id:
            audit.record_global_reset(
                participation_id=participation_id,
                approach_index=int(session.get("current_phase", 0)),
                iteration=int(session.get("iteration", 1)),
                trigger=trigger,
                scope=scope,
                active_model=active_model,
            )
        session["cumulative_adjustments"] = {}
        session["feature_adjustments"] = {}
        session["user_touched_features"] = []
        session["excluded_movies_from_text"] = []
        session["last_text_steering"] = {}
```

## 7. ELSA seed re-weighting from likes

The ELSA seed embedding `\hat{s}` is a weighted mean of two pools: the original elicitation movies `E_0` (weight 1 each, hardcoded) and the participant's current liked movies `L`, capped at the first `like_cap` ids (sorted ascending) and weighted by `λ = selection_signal_weight` per liked movie. `λ` is a config key (per-approach with study-level fallback; default `0.5`, or `0.25` if the approach uses sliders/toggles/hybrid — see `default_selection_signal_weight` in `study_config.py`). `like_cap` is **hardcoded** to `10` at the call site in `iteration_controller.py`.

$$
\hat{s} \;=\; \frac{\sum_{m \in E_0} \text{emb}(m)\;+\;\lambda \cdot \sum_{m \in L^{\le k}} \text{emb}(m)}{|E_0| \;+\; \lambda \cdot |L^{\le k}|}
$$

where `L^{≤k}` is `sorted(L)[:like_cap]` and `|·|` counts only ids that resolve in `recommender.item_ids`. The seed is recomputed from the elicitation pool every iteration — it is **not** a running blend `(1-λ)·\hat{s}_{old} + λ·…`. The seed is then consumed by the recommender as the query vector for the cosine-similarity CF term (see §1.2 and §8 below).

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

## 8. Final ranking (pointer)

The study-level `reranking_strategy` config key is an enum; the only value implemented in this build is `feature-conditioned` (the default). `latent-perturbation` and `constrained-opt` are reserved enum members for future research strategies — selecting them currently falls back to `feature-conditioned`. The math here is just the bookkeeping needed to read a participant's recommendation list:

$$
\text{score}(i) \;=\; \underbrace{\cos(e_i, \hat{s}) \cdot w_{cf}}_{\text{CF}}
\;+\; \underbrace{j_i \cdot w_{g}}_{\text{genre}}
\;+\; \underbrace{\text{clip}\!\big(\gamma \cdot (f_i \cdot a),\; \pm c\big)}_{\text{SAE steering}}
$$

with `e_i` the ELSA item embedding, `\hat{s}` from §7, `f_i` the item's SAE row, `a` the expanded per-neuron profile from §1, and `γ`, `c` chosen **adaptively per iteration** based on the candidate-set's IQR / percentile statistics (not config keys). The blend weights `w_{cf}`, `w_{g}` come from `build_blend_plan(feature_adjustments)` (also derived, not config-driven). The relevant code lives in `recommendation/sae_recommender.py:330-470`; consult that file directly if you need to extend the ranking step.

---

## Appendix: What is configurable

Every numeric constant that appears in a formula below is either a **config key** (researcher can change it per study or per approach) or a **hardcoded** value (intentionally fixed in code). The table below is the single source of truth; inline mentions in §1–§8 only repeat the key name, not the scope.

| Symbol in formulas | Code name | Default | Scope | Read from |
| --- | --- | --- | --- | --- |
| `α` (slider amplification) | `SLIDER_AMPLIFICATION` | `2.0` | **hardcoded** | `modalities/sliders.py` |
| `β` (toggle magnitude) | `toggle_default_weight` | `0.65` | per-approach (study-level fallback) | `active_model` → `conf` |
| `w*` (text baseline) | `text_steering_weight` | `0.55` | per-approach (study-level fallback) | `active_model` → `conf` |
| top-K (text) | `text_steering_top_k` | `6` | per-approach (study-level fallback) | `active_model` → `conf` |
| text composition | `text_composition_mode` ∈ {`replace`, `add`, `intersect`} | `replace` | per-approach (study-level fallback `text_steering.composition_mode`) | `active_model` → `conf["text_steering"]` |
| max query length | `text_steering.max_query_chars` | `200` | study-level | `conf["text_steering"]` |
| `s` (example strength) | `example_selection_weight` | `0.65` | per-approach (study-level fallback) | `active_model` → `conf` |
| top-K (examples) | `example_selection_top_k` | `6` | per-approach (study-level fallback) | `active_model` → `conf` |
| examples merged with sliders? | `use_selected_movies_as_examples` | `false` | per-approach (study-level fallback) | `active_model` → `conf` |
| `λ` (like weight for seed) | `selection_signal_weight` | `0.5` (or `0.25` with sliders/toggles/hybrid) | per-approach (study-level fallback) | `active_model` → `conf` |
| `like_cap` | — | `10` | **hardcoded** at the call site | `iteration_controller.py` |
| SAE checkpoint | `sae` | `DEFAULT_TOPK_SAE_MODEL_ID` | per-approach | `active_model` |
| reranking strategy | `reranking_strategy` | `feature-conditioned` (the only one implemented) | study-level | `conf` |
| `γ`, `c` (clamp) | adaptive | computed per iteration from candidate-set quantiles | **hardcoded** | `recommendation/sae_recommender.py` |
| token-match weights `2.5 / 1.25 / 0.75`, phrase bonus `3.0`, coverage term | — | as shown | **hardcoded** | `modalities/text.py` |
| intensity ladder `{0.65, 1.0, 1.35}`, `_NEGATIVE_HINTS`, `_INTENSITY_*`, `_STOP_WORDS` | — | as shown | **hardcoded** | `modalities/text.py` |
| weight bounds `[0.25, 0.95]`, `[0.1, 0.95]` (text); `[0, 0.95]` (examples); `[-0.95, 0.95]` (text `add` mode) | — | as shown | **hardcoded** | `modalities/text.py`, `modalities/examples.py`, `routes/steering/actions.py` |
| example score boost factor `0.6` | — | `0.6` | **hardcoded** | `modalities/examples.py` |
| drop thresholds `10⁻³` (slider/toggle) and `10⁻⁴` (cluster expansion) | — | as shown | **hardcoded** | `modalities/*.py`, `recommendation/semantic_registry.py` |

"Per-approach (study-level fallback)" means: each approach in `conf["models"]` carries its own value; if it is unset on the approach, `normalize_study_config` copies the study-level value down before the iteration loop reads it. Researchers do not need to set both — the create-study form fills the approach values from the study-level defaults.