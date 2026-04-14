# Server Repository Overview

## 📁 Repository Structure Map

```
server/
├── 📄 Core Application Files (EXISTING)
│   ├── app.py                    # Flask app initialization, plugin manager setup
│   ├── main.py                   # Main blueprint - user study management endpoints
│   ├── auth.py                   # Authentication blueprint (login/signup)
│   ├── models.py                 # SQLAlchemy database models
│   ├── common.py                 # Shared utilities (translations, config loading)
│   └── config.py                 # Configuration (empty, uses env vars)
│
├── 📁 plugins/                   # Plugin system (EXISTING framework)
│   ├── __init__.py               # Plugin registration
│   │
│   ├── utils/                    # ⭐ EXISTING - Shared utilities (212K)
│   │   ├── preference_elicitation.py    # Preference gathering (USED by sae_steering)
│   │   ├── data_loading.py             # Dataset loading (USED by sae_steering)
│   │   ├── interaction_logging.py      # Interaction tracking (USED by sae_steering)
│   │   ├── templates/                  # Shared templates (join, finish, etc.)
│   │   └── static/                     # Shared CSS/JS
│   │
│   ├── fastcompare/              # EXISTING - Fast comparison plugin (68K)
│   │   ├── __init__.py           # Main plugin file (~780 lines)
│   │   ├── algo/                 # Algorithm implementations
│   │   ├── templates/            # UI templates
│   │   └── static/               # Static assets
│   │
│   ├── layoutshuffling/          # EXISTING - Layout shuffling plugin (40K)
│   │   ├── __init__.py           # Main plugin file (~690 lines)
│   │   ├── templates/            # UI templates
│   │   └── static/               # Static assets
│   │
│   ├── vae/                      # EXISTING - VAE wrapper plugin (20K)
│   │   ├── __init__.py           # Minimal plugin (~55 lines)
│   │   └── algorithms.py         # VAE algorithm wrappers
│   │
│   ├── empty_template/           # EXISTING - Template plugin (8K)
│   │   ├── __init__.py           # Minimal template (~71 lines)
│   │   └── templates/            # Example templates
│   │
│   └── sae_steering/             # 🆕 NEW - Our SAE Steering Plugin (384K)
│       ├── __init__.py           # Plugin registration & endpoints
│       ├── sae_recommender.py    # Core SAE recommendation engine
│       ├── text_steering.py      # NLP text-to-feature mapping
│       ├── steering_engine.py    # Feature steering logic
│       ├── sae_integration.py   # SAE model loading & integration
│       ├── templates/            # UI templates (steering interface)
│       ├── static/              # Plugin-specific CSS/JS
│       ├── data/                # Precomputed data (embeddings, indices)
│       ├── models/              # Trained SAE models (.pt files)
│       └── [training scripts]   # train_*.py files
│
├── 📁 templates/                 # EXISTING - Core application templates
│   ├── administration.html
│   ├── login.html
│   └── ...
│
├── 📁 static/                    # EXISTING - Core static assets
│   ├── bootstrap-vue.*
│   ├── datasets/                 # Dataset files
│   └── ...
│
├── 📁 migrations/                 # EXISTING - Database migrations
├── 📁 cache/                     # Runtime cache (generated)
│   └── sae_steering/            # Cache per study GUID
│
└── 📁 instance/                   # EXISTING - Database instance
    └── db.sqlite
```

---

## 🏗️ Main Components

### 1. Core Application (EXISTING - EasyStudy Framework)

**Purpose:** Flask application initialization, plugin management, core infrastructure

**Key Files:**
- `app.py` - Creates Flask app, initializes:
  - SQLAlchemy (database)
  - Flask-PluginKit (plugin manager)
  - Redis (session/cache)
  - Flask-Login (authentication)
  - Flask-Migrate (database migrations)
  
- `main.py` - Core endpoints for:
  - User study management (`/create-user-study`, `/existing-user-studies`)
  - Plugin discovery (`/loaded-plugins`)
  - Participation tracking (`/add-participant`)
  - Results routing (`/results/<plugin>/<guid>`)

- `auth.py` - Authentication endpoints:
  - `/login`, `/logout`, `/signup`

- `models.py` - Database models:
  - `User` - Authenticated users
  - `UserStudy` - Study instances
  - `Participation` - User participation records
  - `Interaction` - User interaction logs

- `common.py` - Shared utilities:
  - `load_languages()` - Translation loading
  - `get_tr()` - Translation function
  - `load_user_study_config()` - Config loading
  - `gen_url_prefix()` - URL generation

### 2. Plugin System (EXISTING - EasyStudy Framework)

**Purpose:** Modular plugin architecture for different study types

**Plugin Registration:**
- Plugins are auto-discovered from `plugins/` directory
- Each plugin must have `__init__.py` with `register()` function
- Plugins register Flask blueprints with prefix `/{plugin_name}`

**Extension Points:**
- **AlgorithmBase** - For recommendation algorithms (not directly used by sae_steering)
- **PreferenceElicitationBase** - For preference gathering (sae_steering uses `utils.preference_elicitation`)
- **Asset Management** - For custom CSS/JS (sae_steering uses standard template/static structure)

**Existing Plugins Overview:**

#### a) fastcompare (68K) - Algorithm Comparison Plugin
- **Purpose:** Compare 2-3 recommendation algorithms side-by-side
- **Size:** ~780 lines in `__init__.py`, largest existing plugin
- **Key Features:**
  - Supports any algorithm implementing `AlgorithmBase`
  - Multiple layout options (rows, columns, single scrollable)
  - Iterative recommendation refinement
  - Comprehensive evaluation metrics
- **Endpoints:** ~15 endpoints (create, join, compare, feedback, results, etc.)
- **Uses:** AlgorithmBase extension point, utils.preference_elicitation, utils.data_loading

#### b) layoutshuffling (40K) - Layout Shuffling Plugin
- **Purpose:** Compare algorithms with shuffled result layouts
- **Size:** ~690 lines in `__init__.py`
- **Key Features:**
  - Compares RLprop with Matrix Factorization
  - Shuffles result layouts between iterations
  - Weight-based recommendation refinement
- **Endpoints:** ~10 endpoints
- **Uses:** utils.preference_elicitation, utils.data_loading

#### c) vae (20K) - VAE Algorithm Wrapper
- **Purpose:** Provides VAE algorithm implementations for fastcompare
- **Size:** ~55 lines in `__init__.py`, minimal plugin
- **Key Features:**
  - Wrapper plugin (doesn't have its own user study flow)
  - Provides VAE algorithms to other plugins
- **Endpoints:** Minimal (just join, initialize)
- **Note:** Not a standalone study plugin, used by fastcompare

#### d) empty_template (8K) - Plugin Template
- **Purpose:** Starting point for creating new plugins
- **Size:** ~71 lines in `__init__.py`
- **Key Features:**
  - Minimal example plugin structure
  - Shows basic plugin registration
  - Template for new plugin development
- **Endpoints:** Basic (join, initialize, dispose)
- **Note:** Not actively used, just a template

#### e) utils (212K) - Shared Utilities Plugin
- **Purpose:** Common functionality for all plugins
- **Size:** Largest plugin by functionality (not a study plugin)
- **Key Modules:**
  - `preference_elicitation.py` - Movie/item selection interface
  - `data_loading.py` - Dataset loading (MovieLens, GoodBooks)
  - `interaction_logging.py` - User interaction tracking
  - `helpers.py` - Utility functions
  - Templates: join, finish, preference_elicitation, etc.
- **Used By:** All study plugins (fastcompare, layoutshuffling, sae_steering)

### 3. Utils Plugin (EXISTING - Shared Utilities)

**Purpose:** Common functionality used by multiple plugins

**Key Modules Used by sae_steering:**
- `preference_elicitation.py` - Movie selection interface, preference gathering
- `data_loading.py` - MovieLens dataset loading (`load_ml_dataset()`)
- `interaction_logging.py` - Logging user interactions (`log_interaction()`, `study_ended()`)

### 4. SAE Steering Plugin (🆕 NEW - Our Contribution)

**Purpose:** SAE-based interpretable and steerable recommendations

**Core Components:**

#### a) Plugin Entry Point (`__init__.py`)
- Registers blueprint with prefix `/sae_steering`
- Defines all HTTP endpoints
- Handles study lifecycle (create, initialize, join, finish)

#### b) SAE Recommender (`sae_recommender.py`)
- **Class:** `SAERecommender`
- **Purpose:** Generates recommendations based on SAE feature adjustments
- **Key Methods:**
  - `load()` - Loads SAE model and precomputed features
  - `get_recommendations(feature_adjustments, n_items, exclude_items)` - Main recommendation method
  - `get_item_features(item_id)` - Get SAE activations for specific item

#### c) Text Steering (`text_steering.py`)
- **Purpose:** Maps natural language to SAE neuron adjustments
- **Key Functions:**
  - `text_to_adjustments(text)` - Converts text to neuron weights
  - `text_to_direct_adjustments(text)` - Direct concept-to-neuron mapping
  - `get_matched_tags(text)` - Returns matched tags for display

#### d) Steering Engine (`steering_engine.py`)
- **Purpose:** Applies feature adjustments to recommendation pipeline
- Handles different steering modes (sliders, toggles, text)

#### e) SAE Integration (`sae_integration.py`)
- **Purpose:** SAE model loading and feature extraction
- Handles different SAE architectures (TopK, Prediction-Aware)

#### f) Templates (`templates/`)
- `sae_steering_create.html` - Study creation interface
- `steering_interface.html` - Main steering UI (sliders, text input, recommendations)
- `sae_steering_results.html` - Results dashboard

---

## 🔌 Integration Points & Endpoints

### Core EasyStudy Endpoints (EXISTING)

**Main Blueprint (`main.py`):**
```
GET  /loaded-plugins                    # List available plugins
GET  /existing-user-studies             # List user studies
POST /create-user-study                 # Create new study
GET  /user-study/<id>                   # Get study details
DELETE /user-study/<id>                 # Delete study
POST /user-study-active                 # Activate/deactivate study
POST /add-participant                   # Add participant to study
GET  /results/<plugin>/<guid>            # Redirect to plugin results
```

**Auth Blueprint (`auth.py`):**
```
GET  /login                             # Login page
POST /login                             # Login handler
GET  /logout                            # Logout
GET  /signup                            # Signup page
POST /signup                            # Signup handler
```

### Existing Plugin Endpoints (for comparison)

**fastcompare Plugin (68K, ~15 endpoints):**
```
GET  /fastcompare/create                          # Study creation
GET  /fastcompare/available-algorithms           # List algorithms
GET  /fastcompare/available-preference-elicitations  # List elicitation methods
GET  /fastcompare/available-data-loaders         # List data loaders
GET  /fastcompare/get-initial-data               # Get items for elicitation
GET  /fastcompare/join?guid=...                  # Join study
GET  /fastcompare/on-joined                      # Post-join callback
GET  /fastcompare/item-search                    # Search items
GET  /fastcompare/send-feedback                  # After preference elicitation
GET  /fastcompare/compare-algorithms             # Main comparison interface
GET  /fastcompare/algorithm-feedback             # Algorithm comparison feedback
GET  /fastcompare/initialize?guid=...            # Initialize study
GET  /fastcompare/finish-user-study              # Complete study
GET  /fastcompare/results?guid=...               # Results dashboard
GET  /fastcompare/fetch-results/<guid>           # Fetch results data
DELETE /fastcompare/dispose?guid=...            # Cleanup
```

**layoutshuffling Plugin (40K, ~12 endpoints):**
```
GET  /layoutshuffling/create                     # Study creation
GET  /layoutshuffling/num-to-select              # Get selection count
GET  /layoutshuffling/join?guid=...              # Join study
GET  /layoutshuffling/on-joined                 # Post-join callback
GET  /layoutshuffling/compare-algorithms         # Main comparison interface
GET  /layoutshuffling/refinement-feedback        # Refinement feedback
GET  /layoutshuffling/algorithm-feedback         # Algorithm feedback
GET  /layoutshuffling/refine-results             # Refine results
GET  /layoutshuffling/send-feedback              # After preference elicitation
GET  /layoutshuffling/initialize?guid=...        # Initialize study
GET  /layoutshuffling/finish-user-study          # Complete study
GET  /layoutshuffling/results                    # Results (minimal)
DELETE /layoutshuffling/dispose                  # Cleanup
```

**vae Plugin (20K, ~2 endpoints):**
```
GET  /vae/join                                   # Join (not supported)
GET  /vae/initialize?guid=...                    # Initialize study
```

**empty_template Plugin (8K, ~3 endpoints):**
```
GET  /emptytemplate/join?guid=...                # Join study
GET  /emptytemplate/initialize?guid=...          # Initialize study
DELETE /emptytemplate/dispose?guid=...          # Cleanup
```

### Existing Plugin Endpoints (for comparison)

**fastcompare Plugin:**
```
GET  /fastcompare/create                          # Study creation
GET  /fastcompare/available-algorithms           # List algorithms
GET  /fastcompare/available-preference-elicitations  # List elicitation methods
GET  /fastcompare/available-data-loaders         # List data loaders
GET  /fastcompare/get-initial-data               # Get items for elicitation
GET  /fastcompare/join?guid=...                  # Join study
GET  /fastcompare/on-joined                      # Post-join callback
GET  /fastcompare/item-search                    # Search items
GET  /fastcompare/send-feedback                  # After preference elicitation
GET  /fastcompare/compare-algorithms             # Main comparison interface
GET  /fastcompare/algorithm-feedback             # Algorithm comparison feedback
GET  /fastcompare/initialize?guid=...            # Initialize study
GET  /fastcompare/finish-user-study              # Complete study
GET  /fastcompare/results?guid=...               # Results dashboard
GET  /fastcompare/fetch-results/<guid>           # Fetch results data
DELETE /fastcompare/dispose?guid=...            # Cleanup
```

**layoutshuffling Plugin:**
```
GET  /layoutshuffling/create                     # Study creation
GET  /layoutshuffling/num-to-select              # Get selection count
GET  /layoutshuffling/join?guid=...              # Join study
GET  /layoutshuffling/on-joined                 # Post-join callback
GET  /layoutshuffling/compare-algorithms         # Main comparison interface
GET  /layoutshuffling/refinement-feedback        # Refinement feedback
GET  /layoutshuffling/algorithm-feedback         # Algorithm feedback
GET  /layoutshuffling/refine-results             # Refine results
GET  /layoutshuffling/send-feedback              # After preference elicitation
GET  /layoutshuffling/initialize?guid=...        # Initialize study
GET  /layoutshuffling/finish-user-study          # Complete study
GET  /layoutshuffling/results                    # Results (minimal)
DELETE /layoutshuffling/dispose                  # Cleanup
```

**vae Plugin:**
```
GET  /vae/join                                   # Join (not supported)
GET  /vae/initialize?guid=...                    # Initialize study
```

**empty_template Plugin:**
```
GET  /emptytemplate/join?guid=...                # Join study
GET  /emptytemplate/initialize?guid=...          # Initialize study
DELETE /emptytemplate/dispose?guid=...          # Cleanup
```

### SAE Steering Plugin Endpoints (🆕 NEW)

**Study Management:**
```
GET  /sae_steering/create               # Study creation page
GET  /sae_steering/available-datasets    # List datasets
GET  /sae_steering/available-sae-models # List SAE models
GET  /sae_steering/available-steering-modes  # List steering modes
GET  /sae_steering/available-neurons    # List available neurons
GET  /sae_steering/initialize?guid=...  # Initialize study
```

**User Participation:**
```
GET  /sae_steering/join?guid=...        # Join study (redirects to utils.join)
GET  /sae_steering/on-joined            # Post-join callback
GET  /sae_steering/get-initial-data     # Get items for preference elicitation
GET  /sae_steering/item-search?pattern=...  # Search items
GET  /sae_steering/show-features        # Show features after elicitation
GET  /sae_steering/steering-interface   # Main steering interface
```

**Steering Operations:**
```
POST /sae_steering/adjust-features      # Apply feature adjustments, get recommendations
GET  /sae_steering/get-recommendations  # Get current recommendations
POST /sae_steering/text-to-adjustments  # Convert text to neuron adjustments
```

**Study Completion:**
```
GET  /sae_steering/finish-user-study    # Complete study
GET  /sae_steering/results?guid=...     # Results dashboard
GET  /sae_steering/fetch-results/<guid> # Fetch results data
DELETE /sae_steering/dispose?guid=...   # Cleanup study data
```

### Integration with Existing Components

**1. Preference Elicitation (EXISTING - Used)**
- **Module:** `plugins.utils.preference_elicitation`
- **Usage:** Called via `utils.preference_elicitation` route
- **Flow:**
  ```
  /sae_steering/join 
    → redirects to /utils/join 
    → redirects to /utils/preference_elicitation
    → continuation_url points to /sae_steering/show-features
  ```
- **Data Flow:** Selected movies stored in `session["elicitation_selected_movies"]`

**2. Data Loading (EXISTING - Used)**
- **Module:** `plugins.utils.data_loading`
- **Function:** `load_ml_dataset()` - Returns `MLDataLoader` instance
- **Usage:** Used in `get_initial_data()`, `item_search()`, recommendation generation

**3. Interaction Logging (EXISTING - Used)**
- **Module:** `plugins.utils.interaction_logging`
- **Functions:**
  - `log_interaction(participation_id, interaction_type, **data)` - Log user actions
  - `study_ended(participation_id)` - Mark study as completed
- **Usage:** Logs feature adjustments, recommendations shown, etc.

**4. Common Utilities (EXISTING - Used)**
- **Module:** `common.py`
- **Functions:**
  - `load_user_study_config(user_study_id)` - Load study configuration
  - `get_tr(languages, lang)` - Translation function
  - `load_languages(base_path)` - Load translation files
  - `multi_lang` - Language decorator

**5. Database Models (EXISTING - Used)**
- **Module:** `models.py`
- **Models Used:**
  - `UserStudy` - Study configuration and state
  - `Participation` - User participation records
  - `Interaction` - User interaction logs

---

## 🔄 Data Flow

### Study Creation Flow
```
Admin → /sae_steering/create
  → Selects dataset, SAE model, steering mode
  → POST /create-user-study (core endpoint)
  → Creates UserStudy record
  → GET /sae_steering/initialize?guid=...
  → Initializes study (loads models, precomputes data)
  → Study ready for participants
```

### User Participation Flow
```
User → /sae_steering/join?guid=...
  → Redirects to /utils/join (demographics)
  → Redirects to /utils/preference_elicitation (movie selection)
  → Uses utils.preference_elicitation module (EXISTING)
  → GET /sae_steering/get-initial-data (provides movies)
  → GET /sae_steering/item-search (search functionality)
  → After selection → /sae_steering/show-features
  → /sae_steering/steering-interface (main UI)
  → POST /sae_steering/adjust-features (steering loop)
  → POST /sae_steering/text-to-adjustments (NLP steering)
  → Iterative refinement
  → /sae_steering/finish-user-study
```

### Recommendation Generation Flow
```
User adjusts features (sliders/text)
  → POST /sae_steering/adjust-features
  → Extracts feature_adjustments from request
  → Calls SAERecommender.get_recommendations()
  → SAERecommender:
    1. Loads SAE model (if not loaded)
    2. Loads precomputed item features
    3. Applies feature_adjustments as weights
    4. Scores items by weighted feature match
    5. Returns top-k recommendations
  → Formats results with movie metadata
  → Returns JSON to frontend
  → Frontend displays recommendations
```

---

## 📊 Key Data Structures

### Session Data (Flask Session)
```python
session["participation_id"]          # Current participation ID
session["user_study_id"]             # Current study ID
session["user_study_guid"]           # Current study GUID
session["elicitation_selected_movies"]  # Selected movies from preference elicitation
session["current_features"]          # Current SAE features displayed
session["feature_adjustments"]       # Current feature adjustments
session["iteration"]                 # Current iteration number
session["cumulative_adjustments"]    # Cumulative adjustments (if cumulative mode)
session["lang"]                      # Current language
```

### UserStudy Configuration (JSON in `settings` field)
```json
{
  "dataset": "ml-latest",
  "sae_model": "prediction_aware_sae",
  "steering_mode": "sliders",
  "num_features_display": 15,
  "num_recommendations": 10,
  "num_iterations": 3,
  "selected_neurons": [42, 17, 89, ...],  // Optional: specific neurons
  "enable_comparison": false,
  "interaction_mode": "reset",  // or "cumulative"
  "models": [...]  // For A/B comparison
}
```

### Feature Adjustments Format
```python
{
  "42": 0.6,   # Neuron ID -> weight (-1.0 to +1.0)
  "17": -0.3,
  "89": 0.8
}
```

---

## 🆕 What's New vs. Existing

### ✅ NEW Components (Our Contribution)

**Plugin:**
- `plugins/sae_steering/` - Entire plugin directory

**Core Logic:**
- `sae_recommender.py` - SAE-based recommendation engine
- `text_steering.py` - Natural language to SAE feature mapping
- `steering_engine.py` - Feature steering logic
- `sae_integration.py` - SAE model integration

**UI:**
- `templates/sae_steering_create.html` - Study creation UI
- `templates/steering_interface.html` - Main steering interface
- `templates/sae_steering_results.html` - Results dashboard
- `static/` - Plugin-specific CSS/JS (if any)

**Data:**
- `data/` - Precomputed embeddings, SAE features, indices
- `models/` - Trained SAE models

**Training Scripts:**
- `train_sae.py` - Basic SAE training
- `train_prediction_aware_sae.py` - Prediction-aware SAE training
- `train_multimodal_sae.py` - Multimodal SAE training
- `train_elsa.py` - ELSA base model training

**Analysis Scripts:**
- `analyze_neurons.py` - Neuron analysis
- `map_tags_to_neurons.py` - Tag-to-neuron mapping
- `build_text_index.py` - Text steering index building

### ⭐ EXISTING Components (Used, Not Modified)

**Core Framework:**
- `app.py`, `main.py`, `auth.py` - Core application
- `models.py` - Database models
- `common.py` - Shared utilities

**Utils Plugin:**
- `plugins/utils/preference_elicitation.py` - **USED** for movie selection
- `plugins/utils/data_loading.py` - **USED** for dataset loading
- `plugins/utils/interaction_logging.py` - **USED** for logging

**Other Plugins:**
- `plugins/fastcompare/` - Existing plugin (68K, ~780 lines, algorithm comparison)
- `plugins/layoutshuffling/` - Existing plugin (40K, ~690 lines, layout shuffling)
- `plugins/vae/` - Existing plugin (20K, ~55 lines, VAE algorithm wrapper)
- `plugins/empty_template/` - Existing plugin (8K, ~71 lines, template only)

**Templates & Static:**
- `templates/administration.html` - Admin interface
- `templates/login.html`, `templates/signup.html` - Auth pages
- `static/` - Core static assets (Bootstrap, Vue, etc.)

---

## 🔗 Key Integration Points

### 1. Plugin Registration
**Location:** `plugins/sae_steering/__init__.py`
```python
def register():
    return {
        "bep": dict(blueprint=bp, prefix=None)
    }
```
- Plugin auto-discovered by Flask-PluginKit
- Blueprint registered with prefix `/sae_steering`

### 2. Preference Elicitation Integration
**Location:** `plugins/sae_steering/__init__.py` (line 356-362)
```python
@bp.route("/on-joined")
def on_joined():
    return redirect(url_for(
        "utils.preference_elicitation",  # EXISTING endpoint
        continuation_url=url_for(f"{__plugin_name__}.show_features"),
        consuming_plugin=__plugin_name__,
        initial_data_url=url_for(f"{__plugin_name__}.get_initial_data"),
        search_item_url=url_for(f"{__plugin_name__}.item_search")
    ))
```

### 3. Data Loading Integration
**Location:** Multiple places in `plugins/sae_steering/__init__.py`
```python
from plugins.utils.data_loading import load_ml_dataset
loader = load_ml_dataset()  # EXISTING function
```

### 4. Interaction Logging Integration
**Location:** `plugins/sae_steering/__init__.py` (line 804-811)
```python
from plugins.utils.interaction_logging import log_interaction, study_ended
log_interaction(
    participation_id,
    "feature-adjustment",  # Interaction type
    iteration=current_iteration,
    adjustments=feature_adjustments,
    ...
)
```

### 5. Configuration Loading
**Location:** Multiple places
```python
from common import load_user_study_config
conf = load_user_study_config(session.get("user_study_id"))  # EXISTING function
```

---

## 📝 Notes

- **No Base Class Extension:** `SAERecommender` is a standalone class, not extending `AlgorithmBase`
- **Direct Endpoint Usage:** Plugin endpoints are called directly, not through extension point interfaces
- **Session-Based State:** User state stored in Flask session, not in database
- **Cache Per Study:** Each study has its own cache directory under `cache/sae_steering/{guid}/`
- **Model Loading:** SAE models loaded lazily on first use, cached in memory
- **Precomputed Features:** Item SAE features precomputed and cached in `data/item_sae_features_*.pt`

---

## 🚀 Quick Start

1. **Start Server:**
   ```bash
   ./start_server.sh
   ```

2. **Access Admin:**
   - Navigate to `/administration`
   - Login/create account

3. **Create Study:**
   - Click "Create" for sae_steering plugin
   - Configure study settings
   - Study will initialize automatically

4. **Join Study:**
   - Use join URL: `/{guid}/sae_steering/join?guid={guid}`
   - Complete preference elicitation
   - Use steering interface

---

## 📊 Plugin Size Comparison

For architecture visualization purposes:

| Plugin | Size | Lines of Code | Purpose | Complexity |
|--------|------|---------------|---------|------------|
| **utils** | 212K | ~2000+ | Shared utilities | High (used by all) |
| **sae_steering** 🆕 | 384K | ~1340 | SAE steering | High (our contribution) |
| **fastcompare** | 68K | ~780 | Algorithm comparison | High |
| **layoutshuffling** | 40K | ~690 | Layout shuffling | Medium |
| **vae** | 20K | ~55 | VAE wrapper | Low |
| **empty_template** | 8K | ~71 | Template | Low |

**Note:** `sae_steering` is the largest plugin due to:
- Precomputed data files (embeddings, SAE features, indices)
- Trained model files (.pt files)
- Multiple training/analysis scripts
- Rich feature extraction and NLP components

---

## 🎨 Architecture Visualization Notes

For drawing architecture diagrams:

### Plugin Hierarchy:
1. **Core Framework** (app.py, main.py, auth.py, models.py)
   - Provides plugin infrastructure
   - Manages user studies, participants, interactions

2. **Utils Plugin** (212K)
   - Foundation layer used by all study plugins
   - Provides preference elicitation, data loading, logging

3. **Study Plugins** (various sizes):
   - **fastcompare** (68K) - Most complex existing plugin
   - **layoutshuffling** (40K) - Medium complexity
   - **sae_steering** (384K) - Our plugin, largest due to ML models/data

4. **Support Plugins**:
   - **vae** (20K) - Algorithm provider
   - **empty_template** (8K) - Template only

### Integration Patterns:
- **All plugins** use `utils.preference_elicitation` for preference gathering
- **All plugins** use `utils.data_loading` for dataset access
- **All plugins** use `utils.interaction_logging` for tracking
- **fastcompare** uses `vae` algorithms via AlgorithmBase
- **sae_steering** is standalone, doesn't extend base classes

### Endpoint Patterns:
- All plugins have: `/create`, `/join`, `/initialize`, `/dispose`
- Study plugins have: preference elicitation flow, main interface, results
- **sae_steering** adds: `/text-to-adjustments`, `/adjust-features` (unique)

---

**Last Updated:** 2025-01-XX
**Author:** Václav Stibor
**Project:** SAE-Based Interpretable Neural Steering for Recommendation Systems

