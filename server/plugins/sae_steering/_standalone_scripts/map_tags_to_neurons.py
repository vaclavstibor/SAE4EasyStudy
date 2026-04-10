"""
Tag-to-Neuron Mapping

Maps movie tags and genres to SAE neurons using:
1. TF-IDF method (paper methodology)
2. Centroid method (simple averaging)

This creates interpretable labels for SAE features.
"""

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pickle
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import entropy

from train_elsa import ELSA, latent_dim
from train_sae import TopKSAE, k, hidden_dim


def has_negation_words(tag):
    """Check for negation words with proper word boundaries."""
    negation_patterns = [
        r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bwithout\b",
        r"\bisn\'t\b", r"\bdoesn\'t\b", r"\bwon\'t\b",
        r"\bcan\'t\b", r"\bdon\'t\b", r"\bdidn\'t\b",
    ]
    return any(re.search(pattern, tag) for pattern in negation_patterns)


def load_data():
    """Load all required data files."""
    tags_df = pd.read_csv("/Users/vaclav.stibor/Library/CloudStorage/OneDrive-HomeCreditInternationala.s/Documents/EasyStudy/server/static/datasets/ml-latest/ml-latest/tags.csv")
    movies_df = pd.read_csv("/Users/vaclav.stibor/Library/CloudStorage/OneDrive-HomeCreditInternationala.s/Documents/EasyStudy/server/static/datasets/ml-latest/ml-latest/movies.csv")
    
    with open("data/item2index.pkl", "rb") as f:
        item2index = pickle.load(f)
    
    index2item = {v: k for k, v in item2index.items()}
    movieid_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))
    
    return tags_df, movies_df, item2index, index2item, movieid_to_title


def load_models(num_items):
    """Load ELSA and SAE models."""
    elsa = ELSA(num_items, latent_dim)
    elsa.load_state_dict(torch.load("models/elsa_model_best.pt"))
    elsa.eval()
    
    sae = TopKSAE(latent_dim, hidden_dim, k)
    sae.load_state_dict(torch.load("models/sae_model_r4_k32.pt"))
    sae.eval()
    
    return elsa, sae


def compute_sparse_embeddings(elsa, sae, num_items, batch_size=1024):
    """Compute sparse SAE representations for all items."""
    print("Computing sparse embeddings...")
    
    with torch.no_grad():
        embeddings = elsa.A.clone()
        h_list = []
        
        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            batch_emb = embeddings[start:end]
            batch_emb_norm = torch.nn.functional.normalize(batch_emb, dim=1)
            _, h_sparse_batch, _ = sae(batch_emb_norm)
            h_list.append(h_sparse_batch)
        
        h_sparse = torch.cat(h_list, dim=0)
    
    print(f"Sparse representations shape: {h_sparse.shape}")
    return h_sparse


def extract_genres(movies_df, item2index):
    """Extract genre-to-items mapping."""
    print("Processing movie genres...")
    
    genre_items = defaultdict(list)
    
    for _, row in movies_df.iterrows():
        movie_id = row["movieId"]
        if movie_id in item2index and pd.notna(row.get("genres")):
            genres = [g.strip().lower() for g in row["genres"].split("|")]
            for genre in genres:
                if genre and genre != "(no genres listed)":
                    genre_items[genre].append(item2index[movie_id])
    
    print(f"Valid genres: {list(genre_items.keys())}")
    return genre_items


def extract_tags(tags_df, item2index, min_occurrences=50):
    """Extract valid tags and their items."""
    print("Processing movie tags...")
    
    tags_df["tag"] = tags_df["tag"].str.lower()
    tag_counts = tags_df["tag"].value_counts()
    
    # Filter tags
    valid_tags = []
    for tag, count in tag_counts.items():
        if (min_occurrences <= count
            and 2 <= len(tag) <= 25
            and not tag.isdigit()
            and (tag.isalpha() or " " in tag or "-" in tag)
            and not has_negation_words(tag)):
            valid_tags.append(tag)
    
    print(f"Valid tags: {len(valid_tags)} (from {len(tag_counts)} total)")
    
    # Create tag-items mapping
    tag_items = defaultdict(list)
    for _, row in tags_df.iterrows():
        movie_id = row["movieId"]
        tag = row["tag"]
        if movie_id in item2index and tag in valid_tags:
            tag_items[tag].append(item2index[movie_id])
    
    return tag_items


def method_tfidf(all_tag_items, sparse_activations):
    """TF-IDF approach following the paper methodology."""
    print("\nMethod: TF-IDF")
    
    all_labels = list(all_tag_items.keys())
    num_labels = len(all_labels)
    num_items = sparse_activations.shape[0]
    num_neurons = sparse_activations.shape[1]
    
    # Build joint distribution matrix [tags x items]
    joint_distribution = torch.zeros(num_labels, num_items)
    for label_idx, (label, items) in enumerate(all_tag_items.items()):
        for item in items:
            joint_distribution[label_idx, item] = 1.0
    
    # Matrix multiplication: [tags x items] @ [items x neurons] = [tags x neurons]
    tag_neuron_matrix = torch.mm(joint_distribution, sparse_activations)
    
    # Normalize by number of items per tag
    tag_counts = joint_distribution.sum(dim=1, keepdim=True)
    tag_neuron_matrix = tag_neuron_matrix / (tag_counts + 1e-8)
    
    # Create documents for TF-IDF
    label_documents = []
    for label_idx in range(num_labels):
        neuron_activations = tag_neuron_matrix[label_idx]
        doc_words = []
        for neuron_idx, activation in enumerate(neuron_activations):
            if activation > 1e-6:
                word_count = max(1, min(50, int(np.sqrt(activation.item()) * 100)))
                doc_words.extend([f"n{neuron_idx}"] * word_count)
        label_documents.append(" ".join(doc_words) if doc_words else "empty")
    
    # Apply TF-IDF
    tfidf = TfidfVectorizer(
        token_pattern=r"n\d+",
        max_features=num_neurons,
        lowercase=False,
        min_df=1,
        max_df=1.0,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(label_documents)
    feature_names = tfidf.get_feature_names_out()
    
    # Convert back to label-neuron mapping
    label_neuron_tfidf = np.zeros((num_labels, num_neurons))
    for label_idx in range(num_labels):
        for feature_idx, tfidf_score in enumerate(tfidf_matrix[label_idx].toarray().flatten()):
            if tfidf_score > 0:
                feature_name = feature_names[feature_idx]
                neuron_idx = int(feature_name[1:])
                if neuron_idx < num_neurons:
                    label_neuron_tfidf[label_idx, neuron_idx] = tfidf_score
    
    # L2 normalize rows
    row_norms = np.linalg.norm(label_neuron_tfidf, axis=1, keepdims=True)
    label_neuron_tfidf = label_neuron_tfidf / (row_norms + 1e-8)
    
    return torch.from_numpy(label_neuron_tfidf).float(), all_labels


def method_centroid(all_tag_items, sparse_activations):
    """Simple centroid prototype method."""
    print("\nMethod: Centroid")
    
    tag_vectors = []
    valid_names = []
    
    for label, items in all_tag_items.items():
        tag_activations = sparse_activations[items]
        centroid = tag_activations.mean(dim=0)
        
        if centroid.norm() > 0:
            centroid = torch.nn.functional.normalize(centroid, dim=0)
            tag_vectors.append(centroid)
            valid_names.append(label)
    
    return torch.stack(tag_vectors) if tag_vectors else torch.empty(0, hidden_dim), valid_names


def compute_metrics(tag_vectors, tag_names):
    """Compute evaluation metrics for tag vectors."""
    if len(tag_vectors) == 0:
        return {"num_tags": 0}
    
    # Tag entropy (lower = more focused)
    tag_entropies = []
    for vector in tag_vectors:
        probs = torch.abs(vector)
        probs = probs / (probs.sum() + 1e-8)
        probs = probs.numpy() + 1e-12
        tag_entropies.append(entropy(probs, base=2))
    
    # Neuron usage
    neuron_usage = (tag_vectors.abs() > 1e-6).any(dim=0).sum().item()
    
    # Sparsity
    sparsity = (tag_vectors.abs() > 1e-6).sum(dim=1).float().mean().item()
    
    # Separation (1 - avg cosine similarity)
    if len(tag_vectors) > 1:
        tag_vectors_norm = torch.nn.functional.normalize(tag_vectors, dim=1)
        similarity_matrix = torch.mm(tag_vectors_norm, tag_vectors_norm.t())
        mask = ~torch.eye(len(tag_vectors), dtype=bool)
        avg_similarity = similarity_matrix[mask].mean().item()
    else:
        avg_similarity = 0
    
    return {
        "avg_entropy": np.mean(tag_entropies),
        "std_entropy": np.std(tag_entropies),
        "neuron_usage": neuron_usage,
        "sparsity": sparsity,
        "separation": 1 - avg_similarity,
        "num_tags": len(tag_vectors),
    }


def main():
    # Load data
    tags_df, movies_df, item2index, index2item, movieid_to_title = load_data()
    num_items = len(item2index)
    
    # Load models
    elsa, sae = load_models(num_items)
    
    # Compute sparse embeddings
    h_sparse = compute_sparse_embeddings(elsa, sae, num_items)
    
    # Extract tags and genres
    genre_items = extract_genres(movies_df, item2index)
    tag_items = extract_tags(tags_df, item2index)
    
    # Combine into single mapping
    all_tag_items = {}
    for tag, items in tag_items.items():
        all_tag_items[f"tag:{tag}"] = items
    for genre, items in genre_items.items():
        all_tag_items[f"genre:{genre}"] = items
    
    print(f"\nTotal labels: {len(all_tag_items)} ({len(tag_items)} tags + {len(genre_items)} genres)")
    
    # Run both methods
    methods = {}
    
    tfidf_vectors, tfidf_names = method_tfidf(all_tag_items, h_sparse)
    methods["tfidf"] = (tfidf_vectors, tfidf_names)
    
    centroid_vectors, centroid_names = method_centroid(all_tag_items, h_sparse)
    methods["centroid"] = (centroid_vectors, centroid_names)
    
    # Evaluate methods
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    results = {}
    for method_name, (vectors, names) in methods.items():
        metrics = compute_metrics(vectors, names)
        results[method_name] = metrics
        
        print(f"\n{method_name.upper()}:")
        print(f"  Tags: {metrics['num_tags']}")
        print(f"  Entropy: {metrics['avg_entropy']:.3f} +/- {metrics['std_entropy']:.3f}")
        print(f"  Neuron usage: {metrics['neuron_usage']}/{hidden_dim}")
        print(f"  Sparsity: {metrics['sparsity']:.1f} neurons/tag")
        print(f"  Separation: {metrics['separation']:.3f}")
    
    # Save results
    print("\n" + "=" * 60)
    print("SAVING")
    print("=" * 60)
    
    for method_name, (vectors, names) in methods.items():
        data = {
            "method_name": method_name,
            "unique_tags": names,
            "tag_tensor": vectors,
            "metrics": results[method_name],
        }
        filename = f"models/tag_neuron_map_{method_name}.pt"
        torch.save(data, filename)
        print(f"Saved: {filename}")
    
    # Save summary
    summary = {
        "available_methods": list(methods.keys()),
        "results": results,
        "genre_count": len(genre_items),
        "tag_count": len(tag_items),
    }
    torch.save(summary, "models/tag_neuron_summary.pt")
    print("Saved: models/tag_neuron_summary.pt")


if __name__ == "__main__":
    main()
