# SAE Steering Plugin

## Overview

This plugin extends EasyStudy to support Sparse Autoencoder (SAE) based steering of neural recommender systems. It enables users to understand and control recommendations through interpretable features extracted from neural network representations.

## Key Features

- **Interpretable Features**: SAE-derived features that represent human-understandable concepts
- **Multiple Steering Modes**: 
  - Sliders: Continuous adjustment of feature strengths
  - Toggles: Binary on/off control
- **Real-time Reranking**: Recommendations update based on feature adjustments
- **Comprehensive Logging**: All steering actions logged for analysis

## Architecture

### Core Components

1. **SAE Integration Layer** (`sae_integration.py`)
   - SAE model loading and management
   - Feature extraction from embeddings
   - Embedding modification based on feature adjustments

2. **Steering Engine** (`steering_engine.py`)
   - Feature adjustment logic
   - Recommendation reranking strategies
   - Impact analysis

3. **Web Interface** (templates/)
   - Study creation interface
   - Interactive steering interface
   - Results dashboard

### Data Flow

1. **Study Creation**: Administrator configures dataset, SAE model, steering mode
2. **Initialization**: Long-running process loads models and precomputes embeddings
3. **Participation**: Users rate items, view features, adjust features, receive recommendations
4. **Analysis**: Results aggregated and analyzed

## Usage

### Creating a Study

1. Navigate to Administration panel
2. Select "SAE Steering" plugin
3. Configure:
   - Dataset (MovieLens, GoodBooks)
   - SAE model
   - Steering mode (sliders/toggles)
   - Number of features to display
   - Number of recommendations
   - Number of iterations

### Participant Flow

1. Join study via invitation link
2. Complete preference elicitation (rate initial items)
3. View active features with descriptions
4. Adjust features using sliders or toggles
5. Receive steered recommendations
6. Repeat for multiple iterations

## Implementation Status

### ✅ Implemented (POC)

- Plugin structure and registration
- Basic web endpoints
- Study creation interface
- Steering interface template
- SAE model wrapper
- Feature extraction logic
- Steering engine with reranking

### 🚧 In Progress

- Dataset integration with EasyStudy data loaders
- Pretrained model loading
- Feature label generation
- Results analysis

### 📋 Planned

- Multiple reranking strategies
- Example-based steering
- Natural language steering
- Advanced feature labeling
- Comprehensive evaluation metrics
- GoodBooks dataset support

## API Endpoints

- `GET /sae_steering/create` - Study creation UI
- `GET /sae_steering/available-datasets` - List datasets
- `GET /sae_steering/available-sae-models` - List SAE models
- `GET /sae_steering/available-steering-modes` - List steering modes
- `GET /sae_steering/initialize` - Start study initialization
- `GET /sae_steering/join` - Participant entry point
- `GET /sae_steering/steering-interface` - Main steering UI
- `POST /sae_steering/adjust-features` - Apply feature adjustments
- `GET /sae_steering/results` - Results dashboard
- `DELETE /sae_steering/dispose` - Cleanup study data

## Database Schema

Reuses existing EasyStudy tables:
- `UserStudy` - Study configuration
- `Participation` - Participant data
- `Interaction` - Logs feature adjustments and steering actions

## Configuration

Study configuration stored in `UserStudy.settings` as JSON:

```json
{
  "dataset": "ml-32m-filtered",
  "sae_model": "simple_sae_64_512",
  "steering_mode": "sliders",
  "num_features": 10,
  "num_recommendations": 20,
  "num_iterations": 3
}
```

## Dependencies

- PyTorch: SAE model implementation
- NumPy: Matrix operations
- Flask: Web framework (existing)
- SQLAlchemy: Database (existing)

## Future Extensions

1. **Advanced Steering Modalities**
   - Example-based: "More like this movie"
   - Natural language: "Show me atmospheric films"
   - Hybrid approaches

2. **Sophisticated Reranking**
   - Latent space perturbation
   - Constrained optimization
   - Multi-objective balancing

3. **Enhanced Interpretability**
   - Automatic feature labeling using LLMs
   - Feature interaction visualization
   - Concept drift detection

4. **Evaluation Infrastructure**
   - Comprehensive questionnaires
   - A/B testing support
   - Longitudinal study capabilities

## Research Applications

This plugin enables research on:
- Effectiveness of interpretable features for user control
- User preferences for different steering modalities
- Impact of transparency on user trust and satisfaction
- Feature interpretability and alignment with user mental models

## Recommendation Steering (Paper Description)

### Online Hybrid Steering Pipeline

The production plugin implements recommendation steering as an online hybrid
ranking pipeline over the `ml-32m-filtered` dataset (\(|\mathcal{I}|=8328\)).
Each interaction step combines three signals:

1. **Collaborative similarity (ELSA)** in dense embedding space.
2. **Genre-overlap prior** derived from elicitation and explicit likes.
3. **Sparse concept steering (TopKSAE)** from cluster sliders.

This design separates *taste matching* (ELSA) from *interpretable control*
(SAE sliders), while preserving stable iteration-to-iteration behavior.

### Elicitation Projection to User Seed

Let \(\mathcal{E}\subset\mathcal{I}\) be elicitation-selected items and
\(\mathbf{e}_i\in\mathbb{R}^{512}\) the ELSA item embedding. The initial user
seed is:

\[
\mathbf{s}_0=\frac{1}{|\mathcal{E}|}\sum_{i\in\mathcal{E}}\mathbf{e}_i.
\]

In parallel, the plugin builds a seed genre set \(G_0\) as the union of
MovieLens genres over \(\mathcal{E}\).

### Like-Based Online Seed Update

At iteration \(t\), users can like displayed items \(\mathcal{L}_t\). The seed
is recomputed from scratch (to support both likes and un-likes cleanly), with
bounded like contribution:

\[
\mathbf{s}_t=
\frac{\sum_{i\in\mathcal{E}}1\cdot\mathbf{e}_i+\sum_{j\in\mathcal{L}_t}\lambda\cdot\mathbf{e}_j}
{|\mathcal{E}|+\lambda|\widetilde{\mathcal{L}}_t|},
\quad \lambda=0.5,\;|\widetilde{\mathcal{L}}_t|\le K_{like}=10.
\]

This yields a smooth adaptation mechanism that keeps elicitation influence
while letting explicit likes progressively steer the profile, without
unbounded drift when users select many items.

Interface-aware calibration is applied in the online system: in slider-based
phases (`sliders`/`both`/`toggles`), likes are down-weighted to
`LIKE_WEIGHT = 0.25`; in non-steering phases (`none`/`text`), likes use
`LIKE_WEIGHT = 0.5`. This separates the explicit control channels and reduces
masking of slider effects in comparative studies.

### Cluster-Level Slider Semantics

Sliders correspond to semantic neuron clusters from offline LLM deduplication.
For cluster \(c\) with neuron set \(N(c)\) and slider delta \(\delta_c\), the
plugin applies equal cluster-level control:

\[
a_k \leftarrow a_k + \delta_c,\;\forall k\in N(c).
\]

Per-request raw slider deltas are scaled by `SLIDER_AMP = 2.0` before
accumulating into the current adjustment vector.

Interpretation note: there is no fixed global conversion from a single slider
value to a fixed number of likes. A slider acts in sparse SAE space
(\(\Delta\mathbf{a}\)) and its final contribution is further modulated by
adaptive \(\gamma_t\) and clamp \(c_t\), while likes modify the dense ELSA
seed \(\mathbf{s}_t\) with interface-specific weights (\(0.25\) in slider
phases, $0.5$ in non-steering phases). Therefore, equivalence between
\"slider = X likes\" is context-dependent and candidate-pool dependent.

For user-facing control, the UI provides per-slider local magnitude hints
(`Low/Medium/High impact`) based on absolute slider value thresholds
(\(|v|<0.35\), \(0.35\le|v|<0.7\), \(|v|\ge0.7\)).

### Slider Lifecycle and Refresh Policy

Let \(\mathcal{C}_t\) denote the currently displayed slider clusters at
iteration \(t\), and let \(\mathcal{T}_t\subseteq\mathcal{C}_t\) be clusters
the user actively moved (non-zero delta in that request).

After each recommendation request, the plugin applies a simple
**ranked-frontier queue** policy:

1. **Steered blacklist.** Any touched cluster \(c\in\mathcal{T}_t\) is added to
   a phase-local steered set and is not auto-shown again.
2. **Shown blacklist.** Any cluster that has already been shown in prior
   iterations of the same phase is not re-shown.
3. **Only-exploit panel policy (all visible sliders).**
   - All slider slots are filled from profile-ranked candidates computed from
     the latest shown recommendation set in the active phase
     (\(\mathcal{R}_t\), top-\(K\) currently shown movies).
   - Candidates are filtered by `not shown`, `not steered`, and
     duplicate-label constraints.
4. **Frontier exhaustion fallback.** If exploit frontier is exhausted, the
   system backfills with global non-steered clusters (safety only, not a fixed
   exploration quota) to avoid empty slider panels.

This mirrors the recommendation-card policy (top-ranked frontier with
seen-item suppression), keeps the UI predictable, and prevents repetitive
re-surfacing of previously steered sliders unless users explicitly request
them (e.g., via search/edit).

### Hybrid Scoring Function

For candidate item \(i\), the final score is:

\[
S_i = s_i^{cf} + s_i^{genre} + s_i^{sae},
\]

with:

\[
s_i^{cf}=\alpha\cdot\cos(\mathbf{e}_i,\mathbf{s}_t),\;\alpha=10,
\]
\[
s_i^{genre}=\beta\cdot\frac{|G(i)\cap G_t|}{|G(i)\cup G_t|},\;\beta=5,
\]
\[
s_i^{sae}=\operatorname{clip}\!\left(\gamma_t\cdot(\mathbf{f}_i^\top\mathbf{a}_t),\;
-c_t,\;c_t\right),
\]

where \(\mathbf{f}_i\in\mathbb{R}^{1024}\) is TopKSAE sparse activation and
\(\mathbf{a}_t\) is the cumulative neuron adjustment vector.

where \(\gamma_t\) is adaptive (computed from candidate-pool scale statistics,
IQR-based) and \(c_t\) is a dynamic clamp derived from the current base-score
span. Operationally, ELSA+genre provides the main relevance signal; SAE acts
as a bounded interpretable modifier for controlled reranking.

### Candidate Filtering and Session Dynamics

Before top-\(K\) extraction (`num_recommendations`, default \(K=20\)), the
pipeline excludes:

- elicitation-selected items,
- already shown items within the same phase,
- explicitly text-suppressed items (if present).

The system supports multi-phase sequential studies (Approach A/B), with
phase-local state for seen items, liked sets, slider touches, and cumulative
adjustments.

### Data Integrity and Runtime Mapping Guarantees

Each returned recommendation is enriched with:

- title and genres (`movies.csv`),
- plot synopsis (`plots.csv`),
- poster URL (`img/{movieId}.jpg`).

A critical deployment constraint is preserving the exact training item-row
ordering. The plugin therefore canonicalizes `item_ids` to training order
(lexicographic `np.unique` over string item IDs from `ratings.csv`) before
scoring. This prevents row-to-item misalignment between stored embeddings,
SAE features, and runtime movie IDs.

In runtime audit on `ml-32m-filtered`, poster and plot coverage is complete
(\(8328/8328\) movies with mapped image and non-empty plot).

### Short-Paper Ready Text Variants

#### Variant A (Conference-Friendly, 3 paragraphs)

We formulate recommendation steering as a hybrid ranking objective that combines collaborative relevance, genre consistency, and bounded concept control. For each candidate item \(i\), the runtime score is \(S_i=s_i^{cf}+s_i^{genre}+s_i^{sae}\), where \(s_i^{cf}\) is cosine similarity in dense ELSA embedding space to the current user seed, \(s_i^{genre}\) is a Jaccard-style genre-overlap bonus, and \(s_i^{sae}\) is a sparse TopKSAE steering term. To avoid unstable re-ranking, the SAE component is scaled by adaptive \(\gamma_t\) (candidate-pool dependent) and dynamically clamped by \(c_t\), so steering remains interpretable and bounded.

User state is updated online from explicit feedback. Elicitation initializes the dense seed, while likes recompute the seed with bounded contribution and cap-based saturation. We apply interface-aware like calibration: lower like weight in slider phases and higher like weight in non-steering phases, preserving fair channel separation in comparative protocols. All state is phase-local (seen items, shown sliders, steered sliders, liked sets), preventing cross-phase leakage in sequential A/B studies.

Slider exposure is controlled by a queue policy aligned with currently shown recommendations. At each step, touched sliders are blacklisted from automatic resurfacing, previously shown sliders are suppressed, and visible sliders are drawn from profile-ranked candidates induced by the current phase recommendation set (\(\mathcal{R}_t\), top-\(K\) shown items). This yields an exploit-only panel policy with predictable behavior; a global non-steered backfill is used only as a safety mechanism when the frontier is exhausted.

#### Variant B (MFF-Friendly, formal style, 3 paragraphs)

Let \(\mathcal{I}\) denote items, \(\mathbf{e}_i\in\mathbb{R}^{512}\) dense ELSA embeddings, \(\mathbf{f}_i\in\mathbb{R}^{1024}\) sparse TopKSAE features, and \(\mathbf{a}_t\) cumulative steering vector at step \(t\). We rank candidates by
\[
S_i(t)=\alpha\cos(\mathbf{e}_i,\mathbf{s}_t)+\beta\cdot\mathrm{Jaccard}(G(i),G_t)+\operatorname{clip}\!\left(\gamma_t\,\mathbf{f}_i^\top\mathbf{a}_t,\,-c_t,\;c_t\right),
\]
with \(\alpha=10,\beta=5\). Here \(\gamma_t\) is adapted from candidate-pool scale statistics (IQR-based), and \(c_t\) is a dynamic bound derived from base-score span, ensuring that sparse steering does not dominate dense relevance.

The seed dynamics are online and bounded. If \(\mathcal{E}\) is elicitation set and \(\mathcal{L}_t\) current likes, \(\mathbf{s}_t\) is recomputed from weighted embeddings of \(\mathcal{E}\cup\mathcal{L}_t\), with capped effective like count to limit drift. In implementation, like weights are interface-conditioned (\(0.25\) in slider phases, \(0.5\) in non-steering phases). This induces a controlled decomposition of explicit feedback channels while keeping longitudinal state stable under add/remove feedback operations.

Slider selection follows a phase-local constrained frontier process. Define \(Z_t\) as shown-slider set and \(U_t\) as steered-slider set; both are monotone within phase. Candidate sliders are filtered by \(c\notin Z_t\), \(c\notin U_t\), and semantic label de-duplication. The visible panel is then filled from profile-ranked clusters computed from the current recommendation context \(\mathcal{R}_t\) (top-\(K\) shown items). Hence, the displayed controls track the active recommendation manifold while preventing repeated exposure of already used controls; global backfill appears only under frontier exhaustion.

## License

Same as EasyStudy project

## Contact

For questions or contributions, contact the research team.
