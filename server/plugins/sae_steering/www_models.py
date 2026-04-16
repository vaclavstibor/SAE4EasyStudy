"""
TopKSAE model architecture and checkpoint loader.

The TopKSAE operates on ELSA item embeddings: it encodes each
L2-normalised embedding into a sparse hidden representation where
only the top-k activations are kept. These sparse activations are
the "features" (neurons) that the steering UI exposes as sliders.

Checkpoint format (produced by train/recsys26):
    {epoch, job_cfg, model_state_dict, optimizer_state_dict}

job_cfg must contain at minimum:
    model_class: "TopKSAE"
    embedding_dim: int   (number of SAE neurons)
    k: int               (top-k sparsity)

input_dim is inferred from encoder_w.shape[0] if absent from job_cfg.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """Top-K Sparse Autoencoder over CF embeddings."""

    def __init__(self, input_dim: int, embedding_dim: int, k: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.k = k
        self.encoder_w = nn.Parameter(torch.empty(input_dim, embedding_dim))
        self.encoder_b = nn.Parameter(torch.zeros(embedding_dim))
        self.decoder_w = nn.Parameter(torch.empty(embedding_dim, input_dim))
        self.decoder_b = nn.Parameter(torch.zeros(input_dim))
        nn.init.kaiming_uniform_(self.encoder_w)
        nn.init.kaiming_uniform_(self.decoder_w)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - x_mean
        x_std = x_centered.std(dim=-1, keepdim=True) + 1e-7
        x_norm = x_centered / x_std
        e_pre = F.relu((x_norm - self.decoder_b) @ self.encoder_w + self.encoder_b)
        topk = torch.topk(e_pre, self.k, dim=-1)
        return torch.zeros_like(e_pre).scatter(-1, topk.indices, topk.values)

    @torch.no_grad()
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse activations for each input row (used by SAERecommender)."""
        return self.encode(x)


def load_checkpoint(filepath: str, device: torch.device = None):
    """Load a TopKSAE checkpoint and return (model, config).

    Handles the train/recsys26 checkpoint format where ``input_dim``
    may be absent from ``job_cfg`` — it is inferred from the state dict.
    """
    device = device or torch.device("cpu")
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    cfg = ckpt["job_cfg"]

    if "input_dim" not in cfg:
        state = ckpt["model_state_dict"]
        cfg["input_dim"] = state["encoder_w"].shape[0]

    model = TopKSAE(
        input_dim=cfg["input_dim"],
        embedding_dim=cfg["embedding_dim"],
        k=cfg.get("k", 32),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


# Backwards-compatible aliases
TopKSAE_WWW = TopKSAE
load_www_checkpoint = load_checkpoint
