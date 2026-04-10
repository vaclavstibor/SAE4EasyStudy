"""
Build Text Index for Multimodal SAE

Creates a direct text-to-neuron mapping using the multimodal SAE,
which understands both collaborative preferences AND content.

This is an improvement over the original approach which used:
1. Text → Tag matching (semantic search)
2. Tag → Neuron mapping (centroid)

New approach:
1. Text → Sentence-BERT embedding
2. Embedding → Multimodal SAE → Direct neuron activations

Usage:
    python build_multimodal_index.py
"""

import json
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR.parent.parent / "static" / "datasets" / "ml-latest"

# Import model definitions
from train_multimodal_sae import (
    MultimodalTopKSAE, 
    ELSA_DIM, TEXT_DIM, COMBINED_DIM, SAE_HIDDEN_DIM, SAE_K
)


def load_multimodal_sae():
    """Load trained multimodal SAE."""
    model = MultimodalTopKSAE(
        elsa_dim=ELSA_DIM,
        text_dim=TEXT_DIM,
        combined_dim=COMBINED_DIM,
        hidden_dim=SAE_HIDDEN_DIM,
        k=SAE_K
    )
    model.load_state_dict(torch.load(MODELS_DIR / "multimodal_sae.pt", map_location='cpu'))
    model.eval()
    return model


def create_concept_embeddings():
    """
    Create embeddings for common movie concepts/queries.
    
    These will be used to create a lookup table for fast inference.
    """
    from sentence_transformers import SentenceTransformer
    
    # Common concepts users might search for
    concepts = {
        # Actors
        "actor:tom cruise": "Tom Cruise movies action star",
        "actor:leonardo dicaprio": "Leonardo DiCaprio dramatic actor oscar",
        "actor:dwayne johnson": "Dwayne Johnson The Rock action comedy",
        "actor:scarlett johansson": "Scarlett Johansson actress marvel",
        "actor:brad pitt": "Brad Pitt actor drama thriller",
        "actor:jennifer lawrence": "Jennifer Lawrence actress hunger games",
        "actor:chris hemsworth": "Chris Hemsworth Thor Marvel action",
        "actor:robert downey jr": "Robert Downey Jr Iron Man Marvel",
        
        # Directors
        "director:christopher nolan": "Christopher Nolan director inception dark knight",
        "director:quentin tarantino": "Quentin Tarantino director pulp fiction",
        "director:martin scorsese": "Martin Scorsese director goodfellas crime",
        "director:steven spielberg": "Steven Spielberg director jurassic park",
        "director:james cameron": "James Cameron director avatar titanic",
        "director:denis villeneuve": "Denis Villeneuve director blade runner dune",
        "director:david fincher": "David Fincher director fight club social network",
        "director:ridley scott": "Ridley Scott director alien gladiator",
        
        # Genres
        "genre:action": "action movies explosions fighting chase",
        "genre:comedy": "comedy funny humor jokes laughing",
        "genre:drama": "drama emotional serious character",
        "genre:horror": "horror scary terrifying monster ghost",
        "genre:sci-fi": "science fiction futuristic space aliens technology",
        "genre:romance": "romance love relationship romantic",
        "genre:thriller": "thriller suspense tension mystery",
        "genre:animation": "animated cartoon pixar disney",
        "genre:documentary": "documentary real life true story",
        "genre:fantasy": "fantasy magic supernatural mythical",
        
        # Themes/Keywords
        "theme:superhero": "superhero marvel dc comics powers",
        "theme:war": "war military battle soldiers combat",
        "theme:crime": "crime gangster mafia heist robbery",
        "theme:family": "family friendly kids children wholesome",
        "theme:adventure": "adventure exploration journey quest",
        "theme:mystery": "mystery detective investigation clues",
        "theme:sports": "sports athletic competition game",
        "theme:music": "music musical singer band concert",
        "theme:historical": "historical period piece history",
        "theme:psychological": "psychological mind twist cerebral",
        
        # Qualities
        "quality:blockbuster": "blockbuster big budget spectacle mainstream",
        "quality:indie": "independent indie art house",
        "quality:classic": "classic old timeless masterpiece",
        "quality:cult": "cult following niche underground",
        "quality:award-winning": "oscar academy award winning critically acclaimed",
        
        # Decades
        "decade:80s": "1980s eighties retro nostalgia",
        "decade:90s": "1990s nineties classic",
        "decade:2000s": "2000s two thousands",
        "decade:2010s": "2010s modern recent",
        "decade:2020s": "2020s new latest current",
        
        # Moods
        "mood:dark": "dark gritty noir bleak",
        "mood:uplifting": "uplifting inspiring feel-good heartwarming",
        "mood:intense": "intense gripping edge-of-seat",
        "mood:relaxing": "relaxing calm peaceful easy",
        "mood:thought-provoking": "thought-provoking philosophical deep meaningful",
        
        # Studios/Franchises
        "franchise:marvel": "Marvel MCU Avengers superhero",
        "franchise:dc": "DC Batman Superman Wonder Woman",
        "franchise:star wars": "Star Wars Luke Skywalker Jedi",
        "franchise:harry potter": "Harry Potter Hogwarts wizards magic",
        "franchise:james bond": "James Bond 007 spy",
        "franchise:pixar": "Pixar animated Toy Story Finding Nemo",
        "franchise:disney": "Disney animated family princess",
    }
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    concept_names = list(concepts.keys())
    concept_texts = list(concepts.values())
    
    print(f"Encoding {len(concepts)} concept embeddings...")
    embeddings = model.encode(concept_texts, show_progress_bar=True, convert_to_tensor=True)
    embeddings = F.normalize(embeddings, dim=1)
    
    return concept_names, embeddings


def build_concept_to_neuron_mapping(sae_model, concept_embeddings, concept_names):
    """
    Map each concept to SAE neurons using the multimodal SAE.
    
    Since we need both ELSA and text embeddings, we use a "concept probe":
    - For text-based concepts, we set ELSA embedding to zeros
    - The SAE will extract the text-relevant features
    """
    
    num_concepts = len(concept_names)
    
    # Create dummy ELSA embeddings (zeros - no preference signal)
    dummy_elsa = torch.zeros(num_concepts, ELSA_DIM)
    
    # Get SAE activations
    with torch.no_grad():
        _, h_sparse, _ = sae_model(dummy_elsa, concept_embeddings)
    
    # Each row of h_sparse is the neuron activation pattern for that concept
    return h_sparse


def build_direct_text_index():
    """
    Build an index that directly maps text queries to neuron adjustments.
    
    This uses the text projection from the multimodal SAE directly.
    """
    
    print("=" * 60)
    print("Building Multimodal Text Index")
    print("=" * 60)
    
    # Load multimodal SAE
    print("\n1. Loading multimodal SAE...")
    try:
        sae_model = load_multimodal_sae()
        print("   ✓ Loaded multimodal SAE")
    except FileNotFoundError:
        print("   ✗ Multimodal SAE not found!")
        print("   Run: python train_multimodal_sae.py first")
        return
    
    # Create concept embeddings
    print("\n2. Creating concept embeddings...")
    concept_names, concept_embeddings = create_concept_embeddings()
    print(f"   ✓ Created {len(concept_names)} concept embeddings")
    
    # Map concepts to neurons
    print("\n3. Mapping concepts to neurons...")
    concept_neurons = build_concept_to_neuron_mapping(
        sae_model, concept_embeddings, concept_names
    )
    print(f"   ✓ Concept-neuron matrix: {concept_neurons.shape}")
    
    # Save index
    print("\n4. Saving index...")
    index_data = {
        "concept_names": concept_names,
        "concept_embeddings": concept_embeddings,
        "concept_neurons": concept_neurons,
        "model_config": {
            "elsa_dim": ELSA_DIM,
            "text_dim": TEXT_DIM,
            "combined_dim": COMBINED_DIM,
            "sae_hidden": SAE_HIDDEN_DIM,
            "sae_k": SAE_K,
        }
    }
    
    torch.save(index_data, DATA_DIR / "multimodal_text_index.pt")
    print(f"   ✓ Saved to: {DATA_DIR / 'multimodal_text_index.pt'}")
    
    # Show examples
    print("\n" + "=" * 60)
    print("Example concept -> neuron mappings:")
    print("=" * 60)
    
    for concept in ["actor:tom cruise", "director:christopher nolan", "genre:action", "genre:sci-fi"]:
        idx = concept_names.index(concept)
        neurons = concept_neurons[idx]
        top_neurons = torch.topk(neurons, 5)
        
        print(f"\n{concept}:")
        for val, neuron_idx in zip(top_neurons.values, top_neurons.indices):
            if val > 0.01:
                print(f"  Neuron {neuron_idx.item():4d}: {val.item():.3f}")
    
    print("\n" + "=" * 60)
    print("Done! Index ready for text steering.")
    print("=" * 60)


if __name__ == "__main__":
    build_direct_text_index()
