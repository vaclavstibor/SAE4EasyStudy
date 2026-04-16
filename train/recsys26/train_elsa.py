import argparse
import ast
import importlib
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim

from datasets import Dataloader, prepare_interaction_data, split_input_target_interactions
from util import (
    evaluate_ndcg_at_k,
    evaluate_recall_at_k,
    get_checkpoint_name,
    get_results_filepath,
    run_training_loop,
    save_results,
    set_seed,
)


def parse_eval_topks(raw_topks: str) -> list[int]:
    return [int(k.strip()) for k in raw_topks.split(",") if k.strip()]


def evaluate_on_split(model, split: sp.csr_matrix, cfg: dict, device: torch.device, topk: int) -> dict:
    inputs, targets = split_input_target_interactions(split, cfg["target_interaction_ratio"])
    inputs, targets = Dataloader(inputs, cfg["batch_size"], device), Dataloader(targets, cfg["batch_size"], device)
    model.eval()
    recalls = evaluate_recall_at_k(model, inputs, targets, topk)
    ndcgs = evaluate_ndcg_at_k(model, inputs, targets, topk)
    return {
        "recall": {"mean": float(np.mean(recalls)), "se": float(np.std(recalls) / np.sqrt(len(recalls)))},
        "ndcg": {"mean": float(np.mean(ndcgs)), "se": float(np.std(ndcgs) / np.sqrt(len(ndcgs)))},
    }


def train_elsa(cfg: dict, device: torch.device):
    print(f"Training ELSA model using config {cfg}")

    _, train_csr, val_csr, test_csr, _, _, _, _ = prepare_interaction_data(cfg)
    eval_topks = parse_eval_topks(cfg["eval_topks"])
    train_dataloader = Dataloader(train_csr, cfg["batch_size"], device, shuffle=True)
    val_dataloader = Dataloader(val_csr, cfg["batch_size"], device)

    model_class = getattr(importlib.import_module(cfg["model_module"]), cfg["model_class"])
    model = model_class(train_csr.shape[1], cfg["embedding_dim"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

    run_training_loop(model, optimizer, train_dataloader, val_dataloader, cfg, device, save_ckpt=ast.literal_eval(os.environ.get("SAVE_CKPT", "True")))

    results = {"val": {f"@{k}": evaluate_on_split(model, val_csr, cfg, device, k) for k in eval_topks}}
    if test_csr.shape[0] > 0:
        results["test"] = {f"@{k}": evaluate_on_split(model, test_csr, cfg, device, k) for k in eval_topks}
    for split, split_res in results.items():
        for k_label, metrics in split_res.items():
            for m in metrics.keys():
                print(f"model = {get_checkpoint_name(cfg)} | split = {split} | {m} {k_label} = {metrics[m]['mean']:.6f} +- {metrics[m]['se']:.6f}")
    save_results(results, cfg, get_results_filepath(cfg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for ELSA training script.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--val_user_ratio", type=float, default=0.2, help="Ratio of validation users")
    parser.add_argument("--test_user_ratio", type=float, default=0.0, help="Ratio of test users")
    parser.add_argument("--target_interaction_ratio", type=float, default=0.2, help="Ratio of interactions used as target")
    parser.add_argument("--model_module", type=str, default="elsa", help="Module containing ELSA model")
    parser.add_argument("--model_class", type=str, default="ELSA", help="Model class name")
    parser.add_argument("--embedding_dim", type=int, required=True, help="Embedding dimension of ELSA model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--early_stopping", type=int, default=10, help="Early stopping number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta_1 coefficient")
    parser.add_argument("--beta2", type=float, default=0.99, help="Adam beta_2 coefficient")
    parser.add_argument("--eval_topk", type=int, default=20, help="Evalutation top k")
    parser.add_argument("--eval_topks", type=str, default="10,20", help="Comma-separated list of k values for reporting")
    parser.add_argument("--seed", type=float, default=42, help="Random seed")
    cfg = vars(parser.parse_args())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
    set_seed(cfg["seed"])
    train_elsa(cfg, device)
