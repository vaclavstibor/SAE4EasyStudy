from copy import deepcopy
from hashlib import sha256
import json
import random
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Tuple
from pathlib import Path

from datasets import Dataloader

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", SCRIPT_DIR))
CHECKPOINT_FOLDER = OUTPUT_ROOT / "checkpoints"
RESULTS_FOLDER = OUTPUT_ROOT / "results"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)  # CPU seed
    torch.mps.manual_seed(seed)  # Metal seed
    torch.cuda.manual_seed(seed)  # GPU seed
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # NumPy seed
    random.seed(seed)  # Python seed


def hash_dict(d: dict, length: int = 8) -> str:
    serialized = json.dumps(d, sort_keys=True).encode()
    return sha256(serialized).hexdigest()[:length]


def get_checkpoint_name(cfg: dict) -> str:
    return f"{cfg['model_class']}-{cfg['embedding_dim']}-{hash_dict(cfg)}"


def get_checkpoint_filepath(cfg: dict) -> str:
    return str(CHECKPOINT_FOLDER / cfg["dataset"] / f"{get_checkpoint_name(cfg)}.ckpt")


def get_results_filepath(cfg: dict) -> str:
    return str(RESULTS_FOLDER / cfg["dataset"] / f"{get_checkpoint_name(cfg)}.json")


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, job_cfg: dict, filepath: str) -> None:
    checkpoint = {
        "epoch": epoch,
        "job_cfg": job_cfg,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)


def save_results(results_dict: dict, job_cfg: dict, filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump({"job_cfg": job_cfg, "results": results_dict}, f, indent=4)


def load_config_from_checkpoint(filepath: str) -> dict:
    return torch.load(filepath, weights_only=False, map_location=torch.device("cpu"))["job_cfg"]


def load_checkpoint(
    model: nn.Module, optimizer: Optional[optim.Optimizer], filepath: str, device: torch.device, job_cfg: Optional[dict] = None
) -> Tuple[int, Optional[dict]]:
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    cfg = checkpoint["job_cfg"]
    if job_cfg is not None and job_cfg != cfg:
        print(f"Loaded checkpoint from {filepath} does not match current job config\nCheckpoint cfg: {checkpoint['job_cfg']}\nCurrent cfg: {job_cfg}")
        print("Starting from scratch.")
        return 0, None
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Loaded checkpoint from {filepath} (after {epoch} epochs)")
    return epoch, cfg


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True)


def run_training_loop(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: Dataloader,
    val_dataloader: Dataloader,
    cfg: dict,
    device: torch.device,
    save_ckpt: bool,
) -> None:
    checkpoint_path = get_checkpoint_filepath(cfg)
    try:
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path, device, cfg)
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")
        start_epoch = 0
    if start_epoch == cfg["epochs"]:
        print(f"Checkpoint already trained for {cfg['epochs']} of {cfg['epochs']} epochs, training is complete.")
        return None

    if cfg["early_stopping"] > 0:
        best_loss, epochs_without_improvement = float("inf"), 0
        best_model_state = None
        best_optimizer_state = None
        best_epoch = start_epoch
    start_time = time.perf_counter()
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{cfg['epochs']}")
        for i, batch in enumerate(pbar):
            loss_dict = model.train_step(optimizer, batch)
            pbar.set_postfix({k: v.item() for k, v in loss_dict.items()})
            if i == len(pbar) - 1:
                model.eval()
                val_losses = {k: 0.0 for k in loss_dict.keys()}
                for val_batch in val_dataloader:
                    batch_losses = model.compute_loss_dict(val_batch)
                    for k, loss_val in batch_losses.items():
                        val_losses[k] += loss_val.item() * val_batch.shape[0] / val_dataloader.dataset_size
                pbar.set_postfix_str(pbar.postfix + " | Val: " + ", ".join([f"{k}={v:.5f}" for k, v in val_losses.items()]))
        if cfg["early_stopping"] > 0:
            if val_losses["Loss"] < best_loss:
                best_loss = val_losses["Loss"]
                best_epoch = epoch
                best_model_state = deepcopy(model.state_dict())
                best_optimizer_state = deepcopy(optimizer.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= cfg["early_stopping"]:
                print("Reached early stopping condition, terminating training.")
                break

    if cfg["early_stopping"] > 0 and best_model_state is not None:
        print(f"Loading best model from epoch {best_epoch + 1} with val loss {best_loss:.5f}.")
        epoch = best_epoch
        model.load_state_dict(best_model_state)
        optimizer.load_state_dict(best_optimizer_state)

    if save_ckpt:
        save_checkpoint(model, optimizer, epoch + 1, cfg, checkpoint_path)
    print(f"Training loop for {get_checkpoint_name(cfg)} took {time.perf_counter() - start_time:.4f} seconds.")


def evaluate_recall_at_k(model, inputs: Dataloader, targets: Dataloader, k: int) -> np.ndarray:
    recall = []
    for input_batch, target_batch in zip(inputs, targets):
        topk_scores, topk_indices = model.recommend(input_batch, k, mask_interactions=True)
        topk_indices = torch.tensor(topk_indices, device=target_batch.device)
        target_batch = target_batch.bool()
        predicted_batch = torch.zeros_like(target_batch).scatter_(1, topk_indices, torch.ones_like(topk_indices, dtype=bool))
        # recall formula from https://arxiv.org/pdf/1802.05814
        r = (predicted_batch & target_batch).sum(axis=1) / torch.minimum(target_batch.sum(axis=1), torch.ones_like(target_batch.sum(axis=1)) * k)
        recall.append(r)
    return torch.cat(recall).detach().cpu().numpy()


def evaluate_ndcg_at_k(model, inputs: Dataloader, targets: Dataloader, k: int) -> np.ndarray:
    ndcg = []
    for input_batch, target_batch in zip(inputs, targets):
        topk_scores, topk_indices = model.recommend(input_batch, k, mask_interactions=True)
        topk_indices = torch.tensor(topk_indices, device=target_batch.device)
        target_batch = target_batch.bool()
        relevance = target_batch.gather(1, topk_indices).float()
        # DCG@k
        gains = 2**relevance - 1
        discounts = torch.log2(torch.arange(2, k + 2, device=relevance.device, dtype=torch.float))
        dcg = (gains / discounts).sum(dim=1)
        # IDCG@k (ideal DCG)
        sorted_relevance, _ = torch.sort(target_batch.float(), dim=1, descending=True)
        ideal_gains = 2 ** sorted_relevance[:, :k] - 1
        ideal_discounts = torch.log2(torch.arange(2, k + 2, device=relevance.device, dtype=torch.float))
        idcg = (ideal_gains / ideal_discounts).sum(dim=1)
        idcg[idcg == 0] = 1
        # nDCG@k
        ndcg.append(dcg / idcg)
    return torch.cat(ndcg).detach().cpu().numpy()


def evaluate_cosine_similarity(model, inputs: Dataloader) -> np.ndarray:
    cosine = []
    for input_batch in inputs:
        output_batch = model(input_batch)[0]
        cosine.append(nn.functional.cosine_similarity(input_batch, output_batch, 1))
    return torch.cat(cosine).detach().cpu().numpy()


def evaluate_l0(model, inputs: Dataloader) -> np.ndarray:
    l0s = []
    for input_batch in inputs:
        e = model.encode(input_batch)[0]
        l0s.append((e > 0).float().sum(-1))
    return torch.cat(l0s).detach().cpu().numpy()


def evaluate_dead_neurons(model, inputs: Dataloader) -> np.ndarray:
    dead_neurons = None
    for input_batch in inputs:
        e = model.encode(input_batch)[0]
        if dead_neurons is None:
            dead_neurons = np.arange(input_batch.shape[1])
        dead_neurons = np.intersect1d(dead_neurons, np.where((e != 0).sum(0).detach().cpu().numpy() == 0)[0])
    return len(dead_neurons)
