"""Generate a paper-ready figure summarising the TopKSAE selection rationale.

Panel (a) reproduces the steering-oriented selection from cell 44 of
training_run_results.ipynb on the full ELSA x TopKSAE validation grid
(gates: relative Recall@20 / nDCG@20 >= 95%, cosine >= 0.80; pick
smallest L0 with quality tie-breakers). Our production checkpoint
(ELSA-512 + TopKSAE-1024, k=32) is the red star.

Panel (b) is a *measured* activation strip for one concrete item under
the selected model: we load the production ELSA-512 and TopKSAE-1024
checkpoints, encode the item embedding, and render the K=1024 concept
vector with its k=32 active cells highlighted. The k in {8,16,64}
rows are intentionally left empty as placeholders for future models
(not yet trained / available locally).

Usage:
    python train/recsys26/results/generate_paper_figure.py
Outputs:
    - train/recsys26/results/figures/paper_topksae_selection.png
    - train/recsys26/results/figures/paper_topksae_selection.pdf
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "ml-32m-filtered"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REPO_ROOT = HERE.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from labeling.labeling_pipeline.model_loader import (  # noqa: E402
    load_elsa_item_embeddings,
    load_sae_checkpoint,
)
from labeling.labeling_pipeline.data_loading import (  # noqa: E402
    load_movies_metadata,
    load_sorted_item_ids_from_ratings,
)

PRODUCTION_CHECKPOINT = "TopKSAE-1024-4d51a427.json"
BACKBONE_INPUT_DIM = 512

# --- Panel (b) configuration ------------------------------------------------
# A canonical item the audience recognises; a different movieId can be set
# via the MOVIE_ID environment variable for ad-hoc regeneration.
DEMO_MOVIE_ID = 2571  # "The Matrix (1999)"
CKPT_DIR = REPO_ROOT / "train/recsys26/checkpoints/ml-32m-filtered"
ELSA_CKPT_PATH = CKPT_DIR / "ELSA-512-c2005bb7.ckpt"
DATASET_DIR = REPO_ROOT / "data_preparation/filters/recsys26/ml-32m-filtered"

# TopKSAE checkpoints for every sparsity shown in panel (b). All four share
# the same ELSA-512 backbone and the same K=1024 SAE width (same concept
# space); only k differs. Missing files render as empty placeholder rows,
# so downloading them from Azure and placing them in CKPT_DIR is enough to
# turn a placeholder into a measured activation.
SAE_CKPT_BY_K = {
    8:  CKPT_DIR / "TopKSAE-1024-b7aad94c.ckpt",
    16: CKPT_DIR / "TopKSAE-1024-f0b0da0a.ckpt",
    32: CKPT_DIR / "TopKSAE-1024-4d51a427.ckpt",  # production
    64: CKPT_DIR / "TopKSAE-1024-77965725.ckpt",
}


def load_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    cf_rows: list[dict] = []
    sae_rows: list[dict] = []
    for fp in sorted(RESULTS.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        if "val" not in d.get("results", {}):
            continue
        cfg = d["job_cfg"]
        val = d["results"]["val"]
        row = {"checkpoint": fp.name, **deepcopy(cfg)}
        if fp.name.startswith("ELSA"):
            topk = "@20" if "@20" in val else "@10"
            for metric in ("recall", "ndcg"):
                payload = val[topk].get(metric, {})
                if isinstance(payload, dict):
                    row[f"{metric} mean"] = payload.get("mean")
                    row[f"{metric} se"] = payload.get("se")
            cf_rows.append(row)
        elif fp.name.startswith("TopKSAE"):
            for m, mv in val.items():
                if isinstance(mv, dict):
                    for k, v in mv.items():
                        row[f"{m} {k}"] = v
                else:
                    row[m] = mv
            row["input_dim"] = int(row["pretrained_model_checkpoint"].split("-")[1])
            sae_rows.append(row)

    cf = pl.DataFrame({k: [r.get(k) for r in cf_rows] for k in cf_rows[0].keys()})
    sae = pl.DataFrame({k: [r.get(k) for r in sae_rows] for k in sae_rows[0].keys()})
    sae = (
        sae
        .with_columns(
            pl.when(pl.col("reconstruction_loss") == "Cosine")
            .then(pl.col("model_class") + " (" + pl.col("reconstruction_loss") + ")")
            .otherwise(pl.col("model_class"))
            .alias("model_class")
        )
        .with_columns((pl.col("embedding_dim") / pl.col("input_dim")).cast(pl.Int32).alias("scaling_factor"))
        .with_columns(
            (100 * pl.col("recall mean") / (pl.col("recall mean") - pl.col("recall degradation mean"))).alias("relative recall"),
            (100 * pl.col("ndcg mean") / (pl.col("ndcg mean") - pl.col("ndcg degradation mean"))).alias("relative ndcg"),
        )
        .filter(pl.col("model_class").str.starts_with("TopKSAE"))
    )
    return cf, sae


def _prep(sae: pl.DataFrame) -> pl.DataFrame:
    # min(relative recall, relative ndcg) collapses the "both >= gate"
    # criterion from the notebook into a single axis.
    return (
        sae.filter(pl.col("pretrained_model_checkpoint") != "ELSA-1024-66e73686.ckpt")
        .with_columns(
            pl.min_horizontal(["relative recall", "relative ndcg"]).alias("min retention")
        )
    )


def _activation_strip_from_values(ax, *, values: np.ndarray,
                                  active_mask: np.ndarray,
                                  color: str, highlight: bool) -> None:
    """Render a strip: grey background with a thin bar at every active cell.

    Using ``vlines`` (instead of imshow on a K-wide raster) guarantees that
    each of the k active concepts is at least one pixel wide on screen and
    therefore clearly visible even when K is very large.
    """
    K = values.shape[0]
    ax.set_facecolor("#ececec")
    active_idx = np.flatnonzero(active_mask)
    ax.vlines(active_idx, ymin=0.0, ymax=1.0,
              colors=color, linewidths=0.9, alpha=0.95)
    ax.set_xlim(-0.5, K - 0.5)
    ax.set_ylim(0.0, 1.0)
    for side in ax.spines.values():
        side.set_color("#d62728" if highlight else "#bdbdbd")
        side.set_linewidth(1.4 if highlight else 0.6)
    ax.set_xticks([])
    ax.set_yticks([])


def _activation_strip_empty(ax) -> None:
    """Render an empty placeholder strip with a hatched grey fill."""
    ax.set_facecolor("#f4f4f4")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor="none", hatch="////",
                               edgecolor="#d0d0d0", linewidth=0))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    for side in ax.spines.values():
        side.set_color("#cccccc")
        side.set_linewidth(0.5)
        side.set_linestyle((0, (2, 2)))
    ax.set_xticks([])
    ax.set_yticks([])


def _canonicalise_title(raw: str) -> str:
    """Turn MovieLens-style \"Matrix, The (1999)\" into \"The Matrix (1999)\"."""
    if not raw:
        return raw
    year = ""
    body = raw
    if body.endswith(")") and "(" in body:
        lp = body.rfind("(")
        year = body[lp:]
        body = body[:lp].strip()
    for art in (", The", ", A", ", An"):
        if body.endswith(art):
            body = art.lstrip(", ") + " " + body[: -len(art)].strip()
            break
    body = body.strip().strip('"').strip()
    return f"{body} {year}".strip()


def _compute_demo_activations_per_k(
    movie_id: int | None = None,
) -> tuple[dict[int, dict], str]:
    """Compute measured TopKSAE activations for a concrete movie at each k.

    Every TopKSAE checkpoint listed in ``SAE_CKPT_BY_K`` is loaded on the
    shared ELSA-512 backbone and evaluated on the demo item. Missing
    checkpoint files are simply skipped (their k row will render as an
    empty placeholder), so dropping freshly downloaded ckpts into
    ``CKPT_DIR`` is enough to populate additional rows.
    """
    import os
    mid = int(movie_id) if movie_id is not None else int(os.environ.get("MOVIE_ID", DEMO_MOVIE_ID))

    item_embeddings = load_elsa_item_embeddings(ELSA_CKPT_PATH)
    item_ids = load_sorted_item_ids_from_ratings(DATASET_DIR)
    id_to_idx = {int(m): i for i, m in enumerate(item_ids)}
    if mid not in id_to_idx:
        raise KeyError(f"movie id {mid} not in sorted ratings item_ids")
    item_emb = item_embeddings[id_to_idx[mid]].unsqueeze(0)

    meta = load_movies_metadata(DATASET_DIR)
    title_raw = meta.get(mid, {}).get("title", f"movieId={mid}") if meta else f"movieId={mid}"
    title = _canonicalise_title(str(title_raw))

    results: dict[int, dict] = {}
    for k, ckpt_path in SAE_CKPT_BY_K.items():
        if not Path(ckpt_path).exists():
            print(f"[panel b] skipping k={k}: checkpoint not found at {ckpt_path}")
            continue
        sae, cfg = load_sae_checkpoint(ckpt_path)
        cfg_k = int(cfg.get("k", k))
        if cfg_k != k:
            print(f"[panel b] warning: ckpt {ckpt_path.name} declares k={cfg_k}, expected {k}")
        activation = sae.get_feature_activations(item_emb)[0]
        values = activation.cpu().numpy().astype(np.float64)
        results[k] = {
            "movie_id": mid,
            "title": title,
            "activation_values": values,
            "active_mask": values > 0.0,
            "l0": int((values > 0.0).sum()),
            "k": cfg_k,
        }
    return results, title


def make_figure(cf: pl.DataFrame, sae: pl.DataFrame) -> plt.Figure:
    all_df = _prep(sae).select(
        ["checkpoint", "input_dim", "embedding_dim", "k", "l0 mean",
         "min retention", "dead neurons", "cosine mean", "relative recall",
         "relative ndcg"]
    ).to_pandas()

    prod_row = all_df[all_df["checkpoint"] == PRODUCTION_CHECKPOINT]
    if prod_row.empty:
        raise RuntimeError(f"Production checkpoint {PRODUCTION_CHECKPOINT} not found in results.")
    prod_l0 = float(prod_row["l0 mean"].iloc[0])
    prod_val = float(prod_row["min retention"].iloc[0])
    prod_dim = int(prod_row["embedding_dim"].iloc[0])
    prod_backbone = int(prod_row["input_dim"].iloc[0])

    plt.rcParams.update({
        "figure.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    fig = plt.figure(figsize=(7.0, 3.4))
    gs = fig.add_gridspec(
        nrows=1, ncols=2, width_ratios=[1.0, 1.1], wspace=0.28,
        left=0.085, right=0.985, top=0.92, bottom=0.30,
    )

    GATE = 95.0

    # --- LEFT PANEL: retention vs L0 across the full ELSA x TopKSAE grid --
    # For every ELSA backbone we keep the best TopKSAE width K at each k
    # (the selection rule prefers min L0, not max K, so at a given k the
    # higher-retention width wins the tie-break). This collapses the
    # 4 (ELSA) x 3 (K) x 4 (k) = 48-point grid into 4 clean curves.
    ax_l = fig.add_subplot(gs[0, 0])
    best_per_backbone = (
        all_df.sort_values(["input_dim", "k", "min retention"], ascending=[True, True, False])
              .groupby(["input_dim", "k"], as_index=False)
              .first()
    )

    backbones = sorted(best_per_backbone["input_dim"].unique().tolist())
    bb_palette = sns.color_palette("colorblind", n_colors=len(backbones))
    bb_color = dict(zip(backbones, bb_palette))

    for bb in backbones:
        sub = best_per_backbone[best_per_backbone["input_dim"] == bb].sort_values("k")
        linestyle = "-" if bb == prod_backbone else "--"
        linewidth = 1.6 if bb == prod_backbone else 1.0
        ax_l.plot(sub["l0 mean"], sub["min retention"], marker="o",
                  linewidth=linewidth, markersize=5, color=bb_color[bb],
                  alpha=0.9, linestyle=linestyle,
                  label=f"ELSA-{bb}")

    ax_l.axhline(100.0, color="grey", linewidth=0.8, linestyle="--")
    ax_l.axhline(GATE, color="#8a8a8a", linewidth=0.8, linestyle=":")
    ax_l.axhspan(GATE, 101.0, color="#4daf4a", alpha=0.07, zorder=0)

    ax_l.scatter([prod_l0], [prod_val], marker="*", s=220,
                 facecolor="#d62728", edgecolor="black", linewidth=0.8, zorder=5)
    ax_l.annotate(f"selected\n($L_0$={prod_l0:.0f}, {prod_val:.1f}%)",
                  xy=(prod_l0, prod_val),
                  xytext=(-10, 16), textcoords="offset points",
                  ha="right", va="bottom",
                  fontsize=7, color="#8a1a1a",
                  arrowprops=dict(arrowstyle="-", color="#8a1a1a", lw=0.6))

    ax_l.set_xscale("log", base=2)
    xticks_l = [8, 16, 32, 64]
    ax_l.set_xticks(xticks_l)
    ax_l.set_xticklabels([f"{int(x)}" for x in xticks_l])
    ax_l.set_xlabel(r"Active neurons per item ($L_0$)")
    ax_l.set_ylabel("min(Relative Recall@20, nDCG@20) [%]")
    ax_l.set_title("(a) Quality retention above the 95% gate", fontsize=9)
    y_min = min(best_per_backbone["min retention"].min(), GATE) - 1.0
    ax_l.set_ylim(y_min, 101.0)
    ax_l.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)

    # --- RIGHT PANEL: measured activations for one concrete item ----------
    # For each sparsity k we look up a TopKSAE-1024 checkpoint (same ELSA
    # backbone, same K=1024 concept space, only k differs). Available
    # checkpoints produce a real measured activation strip; missing ones
    # render as an empty placeholder.
    demo_by_k, demo_title = _compute_demo_activations_per_k()
    K = prod_dim
    ks = sorted(SAE_CKPT_BY_K.keys())
    inner_gs = gs[0, 1].subgridspec(
        nrows=len(ks) + 1, ncols=1,
        height_ratios=[0.95] + [1.0] * len(ks),
        hspace=0.55,
    )
    title_ax = fig.add_subplot(inner_gs[0])
    title_ax.axis("off")
    title_ax.set_title(
        "(b) Per-item control budget   ($K = {:,}$ SAE concepts)".format(K).replace(",", "{,}"),
        fontsize=9, loc="center", pad=2,
    )
    if demo_title:
        title_ax.text(0.5, 0.20,
                      f"measured for \u201c{demo_title}\u201d",
                      transform=title_ax.transAxes,
                      ha="center", va="top", fontsize=7.5, style="italic",
                      color="#404040")

    strip_axes = []
    for i, k in enumerate(ks, start=1):
        ax_k = fig.add_subplot(inner_gs[i])
        demo = demo_by_k.get(k)
        is_selected = (k == int(prod_l0))
        if demo is not None:
            _activation_strip_from_values(
                ax_k, values=demo["activation_values"],
                active_mask=demo["active_mask"],
                color="#d62728" if is_selected else "#4c72b0",
                highlight=is_selected,
            )
            tail = "   \u2190 selected" if is_selected else ""
            label_right = f"{k} / {K} active  ({100.0 * k / K:.1f}%){tail}"
            label_color = "#8a1a1a" if is_selected else "#253c66"
        else:
            _activation_strip_empty(ax_k)
            label_right = f"{k} / {K} active  ({100.0 * k / K:.1f}%)   \u2014 not available"
            label_color = "#8a8a8a"

        ax_k.set_ylabel(f"$k={k}$", rotation=0, fontsize=9, labelpad=14,
                        va="center", ha="right",
                        color=label_color)
        ax_k.text(1.012, 0.5, label_right, transform=ax_k.transAxes,
                  fontsize=7, va="center", ha="left",
                  color=label_color)
        strip_axes.append(ax_k)

    strip_axes[-1].set_xlabel("SAE concept index (1 cell = 1 concept)",
                              fontsize=8, labelpad=4)

    # --- Combined legend for left panel only ------------------------------
    handles_l = []
    for bb in backbones:
        ls = "-" if bb == prod_backbone else "--"
        lw = 1.6 if bb == prod_backbone else 1.0
        handles_l.append(
            Line2D([0], [0], marker="o", linewidth=lw, color=bb_color[bb],
                   linestyle=ls, markersize=5, label=f"ELSA-{bb}")
        )
    shared = [
        Line2D([0], [0], marker="*", color="#d62728", linestyle="",
               markersize=11, markeredgecolor="black",
               label="selected (min $L_0$ above gate)"),
        Line2D([0], [0], color="#8a8a8a", linestyle=":", linewidth=0.8,
               label=f"{GATE:.0f}% steering gate"),
        Line2D([0], [0], color="grey", linestyle="--", linewidth=0.8,
               label="ELSA baseline (100%)"),
    ]
    fig.legend(handles=handles_l + shared,
               loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=7)

    return fig


def main() -> None:
    cf, sae = load_tables()
    fig = make_figure(cf, sae)
    outputs = [
        FIG_DIR / "paper_topksae_selection.png",
        FIG_DIR / "paper_topksae_selection.pdf",
    ]
    # Mirror the rendered figure into the overleaf folder so the paper
    # picks up the freshest version without a manual copy.
    overleaf_fig_dir = REPO_ROOT / "overleaf" / "recsys26" / "figures"
    if overleaf_fig_dir.is_dir():
        outputs.append(overleaf_fig_dir / "paper_topksae_selection.png")
        outputs.append(overleaf_fig_dir / "paper_topksae_selection.pdf")

    for out in outputs:
        kwargs = {"bbox_inches": "tight"}
        if out.suffix == ".png":
            kwargs["dpi"] = 300
        fig.savefig(out, **kwargs)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
