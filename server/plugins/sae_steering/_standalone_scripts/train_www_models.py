"""
Train WWW_disentangling model architectures on EasyStudy's MovieLens data.

Usage:
    # 1. Train base model (ELSA_WWW or MultVAE)
    python train_www_models.py --stage base --base_model elsa --embedding_dim 512

    # 2. Train SAE on top of the base model
    python train_www_models.py --stage sae --base_checkpoint models/www_elsa_512.ckpt \
                               --sae_class TopKSAE --sae_dim 8192 --k 32

    # 3. Extract item features for the recommender
    python train_www_models.py --stage extract --base_checkpoint models/www_elsa_512.ckpt \
                               --sae_checkpoint models/www_TopKSAE_8192.ckpt
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from www_models import (
    ELSA_WWW, MultVAE, BasicSAE, TopKSAE_WWW, l2_normalize,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR.parent.parent / "static" / "datasets" / "ml-latest"


# ============================================================================
# Data loading (reuses EasyStudy's MovieLens data)
# ============================================================================

def load_interaction_matrix():
    """Load MovieLens interactions as a sparse CSR matrix."""
    import pandas as pd

    ratings = pd.read_csv(DATASET_DIR / "ratings.csv")

    # Use implicit feedback: rating >= 4 → positive
    ratings = ratings[ratings["rating"] >= 4.0]

    user_ids = ratings["userId"].unique()
    item_ids = ratings["movieId"].unique()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {m: i for i, m in enumerate(item_ids)}

    rows = [user2idx[u] for u in ratings["userId"]]
    cols = [item2idx[m] for m in ratings["movieId"]]
    vals = np.ones(len(rows), dtype=np.float32)

    mat = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    print(f"Interactions: {mat.nnz}  users={len(user_ids)}  items={len(item_ids)}")
    return mat, user2idx, item2idx


class SparseRowDataset(torch.utils.data.Dataset):
    def __init__(self, csr):
        self.csr = csr

    def __len__(self):
        return self.csr.shape[0]

    def __getitem__(self, idx):
        row = self.csr[idx].toarray().flatten().astype(np.float32)
        return torch.from_numpy(row)


# ============================================================================
# Stage 1 – train base recommender
# ============================================================================

def train_base(args):
    device = _pick_device()
    mat, user2idx, item2idx = load_interaction_matrix()
    num_items = mat.shape[1]

    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "www_item2index.pkl", "wb") as f:
        pickle.dump(item2idx, f)

    if args.base_model == "elsa":
        model = ELSA_WWW(num_items, args.embedding_dim).to(device)
    else:
        model = MultVAE(num_items, [600], args.embedding_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(SparseRowDataset(mat), batch_size=args.batch_size, shuffle=True, num_workers=0)

    best_loss = float("inf")
    job_cfg = {
        "model_class": "ELSA" if args.base_model == "elsa" else "MultVAE",
        "input_dim": num_items,
        "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
        "dataset": "ml-latest",
    }
    if args.base_model == "multvae":
        job_cfg["hidden_dims"] = "600"
        job_cfg["annealing_beta"] = 0.2
        job_cfg["annealing_steps"] = 2000

    ckpt_path = MODELS_DIR / f"www_{args.base_model}_{args.embedding_dim}.ckpt"

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch = batch.to(device)
            losses = model.train_step(optimizer, batch)
            total += losses["Loss"].item()
        avg = total / len(loader)
        print(f"  loss={avg:.6f}")
        if avg < best_loss:
            best_loss = avg
            torch.save({"epoch": epoch + 1, "job_cfg": job_cfg,
                         "model_state_dict": model.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict()}, ckpt_path)
    print(f"Saved → {ckpt_path}")


# ============================================================================
# Stage 2 – train SAE on base-model embeddings
# ============================================================================

def train_sae(args):
    device = _pick_device()
    mat, _, _ = load_interaction_matrix()

    # Load base model
    ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    base_cfg = ckpt["job_cfg"]
    num_items = mat.shape[1]

    if base_cfg["model_class"] == "ELSA":
        base = ELSA_WWW(num_items, base_cfg["embedding_dim"]).to(device)
    else:
        hdims = [int(x) for x in base_cfg.get("hidden_dims", "600").split(",") if x.strip()]
        base = MultVAE(num_items, hdims, base_cfg["embedding_dim"]).to(device)
    base.load_state_dict(ckpt["model_state_dict"])
    base.eval()

    # Compute user embeddings
    loader = DataLoader(SparseRowDataset(mat), batch_size=args.batch_size, num_workers=0)
    embs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding"):
            batch = batch.to(device)
            if base_cfg["model_class"] == "ELSA":
                embs.append(base.encode(batch).cpu())
            else:
                mu, _ = base.encode(batch)
                embs.append(mu.cpu())
    embeddings = torch.cat(embs, dim=0)
    print(f"Embeddings: {embeddings.shape}")

    input_dim = embeddings.shape[1]
    extra = {"k": args.k, "l1_coef": args.l1_coef}
    if args.sae_class == "TopKSAE":
        sae = TopKSAE_WWW(input_dim, args.sae_dim, args.reconstruction_loss, **extra).to(device)
    else:
        sae = BasicSAE(input_dim, args.sae_dim, args.reconstruction_loss, **extra).to(device)

    optimizer = optim.Adam(sae.parameters(), lr=args.lr)
    ds = TensorDataset(embeddings)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    job_cfg = {
        "model_class": args.sae_class,
        "input_dim": input_dim,
        "embedding_dim": args.sae_dim,
        "reconstruction_loss": args.reconstruction_loss,
        "k": args.k, "l1_coef": args.l1_coef,
        "epochs": args.epochs,
        "dataset": "ml-latest",
        "pretrained_model_checkpoint": str(args.base_checkpoint),
    }
    ckpt_path = MODELS_DIR / f"www_{args.sae_class}_{args.sae_dim}.ckpt"
    best_loss = float("inf")

    for epoch in range(args.epochs):
        sae.train()
        total = 0.0
        for (batch,) in tqdm(loader, desc=f"SAE epoch {epoch+1}/{args.epochs}"):
            batch = batch.to(device)
            losses = sae.train_step(optimizer, batch)
            total += losses["Loss"].item()
        avg = total / len(loader)
        print(f"  loss={avg:.6f}")
        if avg < best_loss:
            best_loss = avg
            torch.save({"epoch": epoch + 1, "job_cfg": job_cfg,
                         "model_state_dict": sae.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict()}, ckpt_path)
    print(f"Saved → {ckpt_path}")


# ============================================================================
# Stage 3 – extract item SAE features for the recommender
# ============================================================================

def extract_features(args):
    device = _pick_device()

    # Load base model to get item embeddings
    ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    base_cfg = ckpt["job_cfg"]
    num_items = base_cfg["input_dim"]

    if base_cfg["model_class"] == "ELSA":
        base = ELSA_WWW(num_items, base_cfg["embedding_dim"]).to(device)
    else:
        hdims = [int(x) for x in base_cfg.get("hidden_dims", "600").split(",") if x.strip()]
        base = MultVAE(num_items, hdims, base_cfg["embedding_dim"]).to(device)
    base.load_state_dict(ckpt["model_state_dict"])
    base.eval()

    item_embeddings = base.get_item_embeddings().to(device)

    # Load SAE
    sae_ckpt = torch.load(args.sae_checkpoint, map_location=device, weights_only=False)
    sae_cfg = sae_ckpt["job_cfg"]
    extra = {"k": sae_cfg.get("k", 32), "l1_coef": sae_cfg.get("l1_coef", 0.0)}
    if sae_cfg["model_class"] == "TopKSAE":
        sae = TopKSAE_WWW(sae_cfg["input_dim"], sae_cfg["embedding_dim"],
                           sae_cfg.get("reconstruction_loss", "Cosine"), **extra).to(device)
    else:
        sae = BasicSAE(sae_cfg["input_dim"], sae_cfg["embedding_dim"],
                        sae_cfg.get("reconstruction_loss", "Cosine"), **extra).to(device)
    sae.load_state_dict(sae_ckpt["model_state_dict"])
    sae.eval()

    features = sae.get_feature_activations(item_embeddings)

    with open(DATA_DIR / "www_item2index.pkl", "rb") as f:
        item2idx = pickle.load(f)
    item_ids = [mid for mid, _ in sorted(item2idx.items(), key=lambda x: x[1])]

    sae_name = Path(args.sae_checkpoint).stem
    out_path = DATA_DIR / f"item_sae_features_{sae_name}.pt"
    torch.save({"features": features.cpu(), "item_ids": item_ids}, out_path)
    print(f"Saved {features.shape} features → {out_path}")


# ============================================================================

def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["base", "sae", "extract"], required=True)
    p.add_argument("--base_model", choices=["elsa", "multvae"], default="elsa")
    p.add_argument("--embedding_dim", type=int, default=512)
    p.add_argument("--base_checkpoint", type=str, default=None)
    p.add_argument("--sae_class", choices=["TopKSAE", "BasicSAE"], default="TopKSAE")
    p.add_argument("--sae_dim", type=int, default=8192)
    p.add_argument("--sae_checkpoint", type=str, default=None)
    p.add_argument("--reconstruction_loss", default="Cosine")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--l1_coef", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    if args.stage == "base":
        train_base(args)
    elif args.stage == "sae":
        train_sae(args)
    elif args.stage == "extract":
        extract_features(args)
