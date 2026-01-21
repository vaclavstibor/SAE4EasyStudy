# Steering Neural Recommenders with Sparse Autoencoders

> Fork of [EasyStudy](https://github.com/pdokoupil/EasyStudy) for interactive SAE-based recommendation steering.

Collaborative filtering models learn powerful latent representations, but these embeddings suffer from **representation entanglement** - individual neurons are polysemantic, encoding multiple unrelated concepts simultaneously. This makes the models opaque and difficult to control.

**Sparse Autoencoders (SAE)** offer a solution. By projecting dense embeddings into a high-dimensional sparse space, SAEs learn disentangled features that often align with human-understandable concepts. Users can then directly manipulate these features to **steer** recommendations - boosting or suppressing specific aspects in real-time.

This repository demonstrates the steering capability through an interactive web application, part of the [Sparse4RESS](_) tutorial on sparse representations for recommendation explanations, steering, and segmentation.

## Key Features

The [sae_steering](./server/plugins/sae_steering/) plugin provides:

- **Slider steering** - continuous adjustment of individual SAE neurons
- **Text steering** - natural language queries converted to neuron activations via Sentence-BERT
- **A/B comparison** - side-by-side evaluation of different model configurations
- **Full interaction logging** - all steering actions captured for research analysis

## Quick Start

Requires Python 3.9 and Redis.

```bash
# Setup
cd server
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r pip_requirements.txt

# Start Redis (in separate terminal)
brew install redis && brew services start redis

# Run
flask --debug run
```

Open `http://localhost:5000`, create an SAE Steering study, and explore. Pretrained models and sample data are included.

## EasyStudy Framework

Built on [EasyStudy](https://github.com/pdokoupil/EasyStudy) by [Patrik Dokoupil](mailto:patrik.dokoupil@matfyz.cuni.cz) and [Ladislav Peska](mailto:ladislav.peska@matfyz.cuni.cz). For deployment details, dataset setup, and Docker configuration, refer to the original [documentation](https://github.com/pdokoupil/EasyStudy#readme).
