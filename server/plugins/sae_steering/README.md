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
  "dataset": "ml-latest",
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

## License

Same as EasyStudy project

## Contact

For questions or contributions, contact the research team.
