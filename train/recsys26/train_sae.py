import argparse
import ast
from copy import deepcopy
import importlib
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from datasets import Dataloader, prepare_interaction_data, split_input_target_interactions
from util import (
    CHECKPOINT_FOLDER,
    evaluate_cosine_similarity,
    evaluate_dead_neurons,
    evaluate_l0,
    evaluate_ndcg_at_k,
    evaluate_recall_at_k,
    get_checkpoint_name,
    get_results_filepath,
    load_config_from_checkpoint,
    run_training_loop,
    load_checkpoint,
    save_results,
    set_seed,
)


def evaluate_on_split(cf_model, sae_model, split: sp.csr_matrix, cf_cfg: dict, sae_cfg: dict, device: torch.device) -> dict:
    inputs, targets = split_input_target_interactions(split, cf_cfg["target_interaction_ratio"])
    inputs, targets = Dataloader(inputs, sae_cfg["batch_size"], device), Dataloader(targets, sae_cfg["batch_size"], device)
    cf_model.eval()
    sae_model.eval()
    input_embeddings = Dataloader(
        np.vstack(
            [
                cf_model.encode(batch).detach().cpu().numpy() if cf_cfg["model_class"] == "ELSA" else cf_model.encode(batch)[0].detach().cpu().numpy()
                for batch in inputs
            ]
        ),
        sae_cfg["batch_size"],
        device,
    )

    if cf_cfg["model_class"] == "ELSA":

        def forward_with_sae(self, x: torch.Tensor) -> torch.Tensor:
            return nn.ReLU()(self.decode(sae_model(self.encode(x))[0]) - x)
    elif cf_cfg["model_class"] == "MultVAE":

        def forward_with_sae(self, x: torch.Tensor) -> torch.Tensor:
            return self.decode(sae_model(self.encode(x)[0])[0])

    disentangled_model = deepcopy(cf_model)
    disentangled_model.forward = forward_with_sae.__get__(disentangled_model, disentangled_model.__class__)
    disentangled_model.eval()
    cosines = evaluate_cosine_similarity(sae_model, input_embeddings)
    l0s = evaluate_l0(sae_model, input_embeddings)
    dead_neurons = evaluate_dead_neurons(sae_model, input_embeddings)
    recalls = evaluate_recall_at_k(cf_model, inputs, targets, cf_cfg["eval_topk"])
    recalls_with_sae = evaluate_recall_at_k(disentangled_model, inputs, targets, cf_cfg["eval_topk"])
    recall_degradations = recalls_with_sae - recalls
    ndcgs = evaluate_ndcg_at_k(cf_model, inputs, targets, cf_cfg["eval_topk"])
    ndcgs_with_sae = evaluate_ndcg_at_k(disentangled_model, inputs, targets, cf_cfg["eval_topk"])
    ndcg_degradations = ndcgs_with_sae - ndcgs
    return {
        "cosine": {"mean": float(np.mean(cosines)), "se": float(np.std(cosines) / np.sqrt(len(cosines)))},
        "l0": {"mean": float(np.mean(l0s)), "se": float(np.std(l0s) / np.sqrt(len(l0s)))},
        "dead neurons": dead_neurons,
        "recall": {"mean": float(np.mean(recalls_with_sae)), "se": float(np.std(recalls_with_sae) / np.sqrt(len(recalls_with_sae)))},
        "recall degradation": {"mean": float(np.mean(recall_degradations)), "se": float(np.std(recall_degradations) / np.sqrt(len(recall_degradations)))},
        "ndcg": {"mean": float(np.mean(ndcgs_with_sae)), "se": float(np.std(ndcgs_with_sae) / np.sqrt(len(ndcgs_with_sae)))},
        "ndcg degradation": {"mean": float(np.mean(ndcg_degradations)), "se": float(np.std(ndcg_degradations) / np.sqrt(len(ndcg_degradations)))},
    }


def train_sae(cfg: dict, device: torch.device):
    print(f"Training sparse autoencoder using config {cfg}")

    cf_model_checkpoint = f"{CHECKPOINT_FOLDER}/{cfg['dataset']}/{cfg['pretrained_model_checkpoint']}"
    cf_model_cfg = load_config_from_checkpoint(cf_model_checkpoint)
    print(f"Source model config: {cf_model_cfg}")
    if cf_model_cfg["model_class"] != "ELSA":
        raise ValueError(f"Unsupported source model '{cf_model_cfg['model_class']}'. Only ELSA checkpoints are supported.")

    _, train_csr, val_csr, test_csr, _, _, _, _ = prepare_interaction_data(cf_model_cfg)
    train_interaction_dataloader = Dataloader(train_csr, cf_model_cfg["batch_size"], device)
    val_interaction_dataloader = Dataloader(val_csr, cf_model_cfg["batch_size"], device)
    cf_model_class = getattr(importlib.import_module(cf_model_cfg["model_module"]), cf_model_cfg["model_class"])
    cf_model = cf_model_class(train_csr.shape[1], cf_model_cfg["embedding_dim"]).to(device)
    load_checkpoint(cf_model, None, cf_model_checkpoint, device)
    train_user_embeddings = np.vstack(
        [
            cf_model.encode(batch).detach().cpu().numpy() if cf_model_cfg["model_class"] == "ELSA" else cf_model.encode(batch)[0].detach().cpu().numpy()
            for batch in tqdm(train_interaction_dataloader, desc="Computing user embeddings from train interactions")
        ]
    )
    val_user_embeddings = np.vstack(
        [
            cf_model.encode(batch).detach().cpu().numpy() if cf_model_cfg["model_class"] == "ELSA" else cf_model.encode(batch)[0].detach().cpu().numpy()
            for batch in tqdm(val_interaction_dataloader, desc="Computing user embeddings from val interactions")
        ]
    )
    print(f"Train user embeddings shape={train_user_embeddings.shape}, val user embeddings shape={val_user_embeddings.shape}")
    train_embedding_dataloader = Dataloader(train_user_embeddings, cfg["batch_size"], device, shuffle=True)
    val_embedding_dataloader = Dataloader(val_user_embeddings, cfg["batch_size"], device)

    sae_model_class = getattr(importlib.import_module(cfg["model_module"]), cfg["model_class"])
    sae_extra_params = {k: cfg[k] for k in cfg.keys() if k in ["l1_coef", "k"]}
    sae_model = sae_model_class(train_user_embeddings.shape[1], cfg["embedding_dim"], cfg["reconstruction_loss"], **sae_extra_params).to(device)
    optimizer = optim.Adam(sae_model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

    run_training_loop(
        sae_model, optimizer, train_embedding_dataloader, val_embedding_dataloader, cfg, device, save_ckpt=ast.literal_eval(os.environ.get("SAVE_CKPT", "True"))
    )

    results = {"val": evaluate_on_split(cf_model, sae_model, val_csr, cf_model_cfg, cfg, device)}
    if test_csr.shape[0] > 0:
        results["test"] = evaluate_on_split(cf_model, sae_model, test_csr, cf_model_cfg, cfg, device)
    for split, split_res in results.items():
        for m in split_res.keys():
            if m in ["recall", "recall degradation", "ndcg", "ndcg degradation"]:
                print(
                    f"model = {get_checkpoint_name(cfg)} | split = {split} | {m} @ {cf_model_cfg['eval_topk']} = {split_res[m]['mean']:.6f} +- {split_res[m]['se']:.6f}"
                )
            elif m in ["cosine", "l0"]:
                print(f"model = {get_checkpoint_name(cfg)} | split = {split} | {m} = {split_res[m]['mean']:.6f} +- {split_res[m]['se']:.6f}")
            else:
                print(f"model = {get_checkpoint_name(cfg)} | split = {split} | {m} = {split_res[m]}")
    save_results(results, cfg, get_results_filepath(cfg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for SAE training script.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--pretrained_model_checkpoint", type=str, required=True, help="Filename of checkpoint containing pre-trained model")
    parser.add_argument("--model_module", type=str, default="sae", help="Module containing SAE model")
    parser.add_argument("--model_class", type=str, default="TopKSAE", help="Model class name")
    parser.add_argument("--embedding_dim", type=int, required=True, help="Embedding dimension of SAE model")
    parser.add_argument("--reconstruction_loss", type=str, default="L2", help="Reconstruction loss (L2 or Cosine)")
    parser.add_argument("--l1_coef", type=float, default=0.01, help="L1 loss coefficient (BasicSAE, TopKSAE)")
    parser.add_argument("--k", type=int, default=32, help="Top K parameter (TopKSAE)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--early_stopping", type=int, default=10, help="Early stopping number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta_1 coefficient")
    parser.add_argument("--beta2", type=float, default=0.99, help="Adam beta_2 coefficient")
    parser.add_argument("--seed", type=float, default=42, help="Random seed")
    cfg = vars(parser.parse_args())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
    set_seed(cfg["seed"])
    train_sae(cfg, device)
