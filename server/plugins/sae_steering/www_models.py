"""
Model architectures from WWW_disentangling project.

Ported for use in EasyStudy's SAE steering pipeline.
Three model families:
  - SAE (BasicSAE, TopKSAE) - sparse autoencoders over CF embeddings
  - ELSA - linear shallow autoencoder (base recommender)
  - MultVAE - multinomial variational autoencoder (base recommender)

The SAE models operate on embeddings produced by a base recommender (ELSA or
MultVAE).  The steering pipeline only needs get_feature_activations() from SAE.
"""

from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + 1e-8)


# ============================================================================
# SAE family (sparse autoencoders for interpretable feature extraction)
# ============================================================================

class SAE(nn.Module):
    """
    Base SAE from WWW_disentangling.

    Key differences from EasyStudy's TopKSAE / PredictionAwareSAE:
    - Uses raw nn.Parameter (not nn.Linear) for encoder/decoder weights
    - Per-sample input standardisation (zero-mean, unit-variance)
    - Decoder columns kept L2-normalised after each step
    """

    def __init__(self, input_dim: int, embedding_dim: int, reconstruction_loss: str = "Cosine"):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.reconstruction_loss = reconstruction_loss
        self.encoder_w = nn.Parameter(nn.init.kaiming_uniform_(torch.empty([input_dim, embedding_dim])))
        self.encoder_b = nn.Parameter(torch.zeros(embedding_dim))
        self.decoder_w = nn.Parameter(nn.init.kaiming_uniform_(torch.empty([embedding_dim, input_dim])))
        self.decoder_b = nn.Parameter(torch.zeros(input_dim))
        self.normalize_decoder()

    @abstractmethod
    def post_process_embedding(self, e: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def total_loss(self, partial_losses: dict) -> torch.Tensor:
        raise NotImplementedError

    # --- forward / encode / decode ------------------------------------------

    def encode(self, x: torch.Tensor):
        x, x_mean, x_std = self.standardize_input(x)
        e_pre = F.relu((x - self.decoder_b) @ self.encoder_w + self.encoder_b)
        return self.post_process_embedding(e_pre), e_pre, x_mean, x_std

    def decode(self, e: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> torch.Tensor:
        return self.destandardize_output(e @ self.decoder_w + self.decoder_b, x_mean, x_std)

    def forward(self, x: torch.Tensor):
        e, e_pre, x_mean, x_std = self.encode(x)
        out = self.decode(e, x_mean, x_std)
        return out, e, e_pre, x_mean, x_std

    # --- normalisation helpers ----------------------------------------------

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder_w.data = l2_normalize(self.decoder_w.data)
        if self.decoder_w.grad is not None:
            self.decoder_w.grad -= (
                (self.decoder_w.grad * self.decoder_w.data).sum(-1, keepdim=True) * self.decoder_w.data
            )

    def standardize_input(self, x: torch.Tensor):
        x_mean = x.mean(dim=-1, keepdim=True)
        x = x - x_mean
        x_std = x.std(dim=-1, keepdim=True) + 1e-7
        x = x / x_std
        return x, x_mean, x_std

    def destandardize_output(self, out: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> torch.Tensor:
        return x_mean + out * x_std

    # --- training -----------------------------------------------------------

    def _compute_loss_dict(self, x, e_pre, e, x_out):
        losses = {
            "L2": (x_out - x).pow(2).mean(),
            "L1": e.abs().sum(-1).mean(),
            "L0": (e > 0).float().sum(-1).mean(),
            "Cosine": (1 - F.cosine_similarity(x, x_out, 1)).mean(),
        }
        losses["Loss"] = self.total_loss(losses)
        return losses

    def compute_loss_dict(self, batch: torch.Tensor) -> dict:
        out, e, e_pre, _, _ = self(batch)
        return self._compute_loss_dict(batch, e_pre, e, out)

    def train_step(self, optimizer: optim.Optimizer, batch: torch.Tensor) -> dict:
        losses = self.compute_loss_dict(batch)
        optimizer.zero_grad()
        losses["Loss"].backward()
        self.normalize_decoder()
        optimizer.step()
        return losses

    # --- EasyStudy compatibility --------------------------------------------

    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse activations for each input row (used by SAERecommender)."""
        with torch.no_grad():
            e, _, _, _ = self.encode(x)
        return e


class BasicSAE(SAE):
    def __init__(self, input_dim: int, embedding_dim: int, reconstruction_loss: str = "Cosine", **extra):
        super().__init__(input_dim, embedding_dim, reconstruction_loss)
        self.l1_coef = extra.get("l1_coef", 1e-3)

    def post_process_embedding(self, e):
        return e

    def total_loss(self, partial_losses):
        return partial_losses[self.reconstruction_loss] + self.l1_coef * partial_losses["L1"]


class TopKSAE_WWW(SAE):
    """TopK variant – keeps only the k largest activations per sample."""

    def __init__(self, input_dim: int, embedding_dim: int, reconstruction_loss: str = "Cosine", **extra):
        super().__init__(input_dim, embedding_dim, reconstruction_loss)
        self.l1_coef = extra.get("l1_coef", 0.0)
        self.k = extra.get("k", 32)

    def post_process_embedding(self, e):
        topk = torch.topk(e, self.k, dim=-1)
        return torch.zeros_like(e).scatter(-1, topk.indices, topk.values)

    def total_loss(self, partial_losses):
        return partial_losses[self.reconstruction_loss] + self.l1_coef * partial_losses["L1"]


# ============================================================================
# Base recommender: ELSA
# ============================================================================

class ELSA_WWW(nn.Module):
    """
    Scalable Linear Shallow Autoencoder for collaborative filtering.

    Paper: https://dl.acm.org/doi/abs/10.1145/3523227.3551482

    Key difference from EasyStudy's ELSA:
    - forward() returns ReLU(decode(encode(x)) - x)  (residual + ReLU)
    - Uses normalised MSE loss
    - Single nn.Parameter ``encoder`` with L2-normalised columns
    """

    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.encoder = nn.Parameter(nn.init.kaiming_uniform_(torch.empty([input_dim, embedding_dim])))
        self.normalize_encoder()

    @torch.no_grad()
    def normalize_encoder(self):
        self.encoder.data = l2_normalize(self.encoder.data)
        if self.encoder.grad is not None:
            self.encoder.grad -= (
                (self.encoder.grad * self.encoder.data).sum(-1, keepdim=True) * self.encoder.data
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.encoder

    def decode(self, e: torch.Tensor) -> torch.Tensor:
        return e @ self.encoder.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.decode(self.encode(x)) - x)

    def get_item_embeddings(self) -> torch.Tensor:
        """Return L2-normalised item embedding matrix."""
        return l2_normalize(self.encoder.data.detach())

    @torch.no_grad()
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Return ReLU(encode(x)) so ELSA can be used as a steering model.

        For SAE models, feature activations are the sparse hidden layer.
        For ELSA there is no sparsity constraint, so we apply ReLU to the
        raw embeddings to get non-negative "activations" that the steering
        pipeline can manipulate in the same way.
        """
        return F.relu(self.encode(x))

    # --- training -----------------------------------------------------------

    def compute_loss_dict(self, batch: torch.Tensor) -> dict:
        pred = self(batch)
        loss = (l2_normalize(pred) - l2_normalize(batch)).pow(2).sum(-1).mean()
        return {"Loss": loss}

    def train_step(self, optimizer: optim.Optimizer, batch: torch.Tensor) -> dict:
        losses = self.compute_loss_dict(batch)
        optimizer.zero_grad()
        losses["Loss"].backward()
        self.normalize_encoder()
        optimizer.step()
        return losses

    @torch.no_grad()
    def recommend(self, interaction_batch: torch.Tensor, k: int, mask_interactions: bool = True):
        scores = self(interaction_batch)
        if mask_interactions:
            scores = torch.where(interaction_batch != 0, 0, scores)
        topk_scores, topk_indices = torch.topk(scores, k)
        return topk_scores.cpu().numpy(), topk_indices.cpu().numpy()


# ============================================================================
# Base recommender: MultVAE
# ============================================================================

class _Linear(nn.Module):
    """Thin wrapper so the checkpoint state_dict keys stay compatible."""
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MultVAE(nn.Module):
    """
    Multinomial Variational Autoencoder for collaborative filtering.

    Paper: https://arxiv.org/pdf/1802.05814
    """

    def __init__(self, input_dim: int, hidden_dims: list, embedding_dim: int,
                 annealing_beta: float = 0.2, annealing_steps: int = 2000):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        dims = [input_dim] + hidden_dims + [embedding_dim]
        self.encoder = nn.Sequential()
        for i in range(len(dims) - 2):
            w = nn.init.xavier_uniform_(torch.empty([dims[i + 1], dims[i]]))
            self.encoder.add_module(f"fc{i}", _Linear(w, torch.zeros(dims[i + 1])))
            self.encoder.add_module(f"act{i}", nn.Tanh())
        self.encoder_mu = _Linear(
            nn.init.xavier_uniform_(torch.empty([dims[-1], dims[-2]])), torch.zeros(dims[-1])
        )
        self.encoder_logvar = _Linear(
            nn.init.xavier_uniform_(torch.empty([dims[-1], dims[-2]])), torch.zeros(dims[-1])
        )

        dims = dims[::-1]
        self.decoder = nn.Sequential()
        for i in range(len(dims) - 1):
            w = nn.init.xavier_uniform_(torch.empty([dims[i + 1], dims[i]]))
            self.decoder.add_module(f"fc{i}", _Linear(w, torch.zeros(dims[i + 1])))
            if i != len(dims) - 2:
                self.decoder.add_module(f"act{i}", nn.Tanh())

        self.annealing_beta = annealing_beta
        self.annealing_steps = annealing_steps
        self.beta = 0.0

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.encoder_mu(h), self.encoder_logvar(h)

    def sample_from_prior(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * logvar.exp().sqrt()

    def decode(self, e: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.decoder(e), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        if self.training:
            return self.decode(self.sample_from_prior(mu, logvar))
        return self.decode(mu)

    @torch.no_grad()
    def get_item_embeddings(self, batch_size: int = 512) -> torch.Tensor:
        """
        Compute item embeddings by encoding one-hot item vectors.

        Returns shape (input_dim, embedding_dim) so the SAE pipeline can
        treat them the same way as ELSA item embeddings.
        """
        self.eval()
        parts = []
        for start in range(0, self.input_dim, batch_size):
            end = min(start + batch_size, self.input_dim)
            one_hot = torch.zeros(end - start, self.input_dim,
                                  device=next(self.parameters()).device)
            for i in range(end - start):
                one_hot[i, start + i] = 1.0
            mu, _ = self.encode(one_hot)
            parts.append(mu)
        return l2_normalize(torch.cat(parts, dim=0))

    # --- training -----------------------------------------------------------

    def compute_loss_dict(self, batch: torch.Tensor) -> dict:
        mu, logvar = self.encode(F.dropout(batch, 0.5))
        out = self.decode(self.sample_from_prior(mu, logvar))
        nll = -torch.mean(batch * torch.log(torch.clamp(out, min=1e-7)), dim=-1)
        d_kl = torch.mean(-0.5 * (1 + logvar - mu.pow(2) - torch.exp(logvar)), dim=-1)
        return {"NLL": torch.mean(nll), "D_KL": torch.mean(d_kl),
                "Loss": torch.mean(nll + self.beta * d_kl)}

    def train_step(self, optimizer: optim.Optimizer, batch: torch.Tensor) -> dict:
        losses = self.compute_loss_dict(batch)
        optimizer.zero_grad()
        losses["Loss"].backward()
        optimizer.step()
        self.beta = min(self.beta + 1 / self.annealing_steps, self.annealing_beta)
        return losses

    @torch.no_grad()
    def recommend(self, interaction_batch: torch.Tensor, k: int, mask_interactions: bool = True):
        scores = self(interaction_batch)
        if mask_interactions:
            scores = torch.where(interaction_batch != 0, 0, scores)
        topk_scores, topk_indices = torch.topk(scores, k)
        return topk_scores.cpu().numpy(), topk_indices.cpu().numpy()


# ============================================================================
# Factory helpers
# ============================================================================

MODEL_REGISTRY = {
    "BasicSAE": BasicSAE,
    "TopKSAE": TopKSAE_WWW,
    "ELSA": ELSA_WWW,
    "MultVAE": MultVAE,
}


def create_model_from_config(cfg: dict) -> nn.Module:
    """Instantiate a model from a WWW_disentangling-style config dict (``job_cfg``)."""
    cls_name = cfg["model_class"]
    cls = MODEL_REGISTRY.get(cls_name)
    if cls is None:
        raise ValueError(f"Unknown model class '{cls_name}'. Available: {list(MODEL_REGISTRY.keys())}")

    if cls_name in ("BasicSAE", "TopKSAE"):
        extra = {k: cfg[k] for k in ("l1_coef", "k") if k in cfg}
        return cls(
            input_dim=cfg.get("input_dim", cfg.get("embedding_dim")),
            embedding_dim=cfg["embedding_dim"],
            reconstruction_loss=cfg.get("reconstruction_loss", "Cosine"),
            **extra,
        )
    elif cls_name == "ELSA":
        return cls(cfg["input_dim"], cfg["embedding_dim"])
    elif cls_name == "MultVAE":
        hidden_dims = cfg.get("hidden_dims", [])
        if isinstance(hidden_dims, str):
            hidden_dims = [int(x) for x in hidden_dims.split(",") if x.strip()]
        return cls(
            cfg["input_dim"], hidden_dims, cfg["embedding_dim"],
            annealing_beta=cfg.get("annealing_beta", 0.2),
            annealing_steps=cfg.get("annealing_steps", 2000),
        )
    raise ValueError(f"Cannot build model for class '{cls_name}'")


def load_www_checkpoint(filepath: str, device: torch.device = None):
    """
    Load a WWW_disentangling checkpoint and return (model, config).

    Checkpoint format: {epoch, job_cfg, model_state_dict, optimizer_state_dict}
    """
    device = device or torch.device("cpu")
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    cfg = ckpt["job_cfg"]
    model = create_model_from_config(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
