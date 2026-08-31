"""
Figure generation for the revised manuscript.

Produces (into GISRNet_Results/figures):

  fig_scatter_train_test        GICQI vs mean DSI, training and test partitions
  fig_scatter_per_fold          test scatter for each of the six folds
  fig_iqa_vs_dsi                LOE / AGIC / GICQI vs mean DSI  (revised Fig. 8)
  fig_correlation_bars          PLCC / SROCC / KROCC per IQA metric
  fig_learning_curves           training loss and validation RMSE per fold
  fig_bland_altman              agreement between GICQI and mean DSI
  fig_residuals                 residual vs predicted, plus residual histogram
  fig_external_scatter          HF_Test and MED-NODE generalisation scatter
  fig_ablation                  head variant / augmentation / TL-vs-RI ablation
  fig_per_method_error          prediction error grouped by correction algorithm
  fig_leakage_diagram           source-disjoint split schematic
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import config as C
from metrics import all_metrics, plcc, srocc, krocc

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 300, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

PRED_DIR = C.OUT_DIR / "predictions"
BLUE, RED, GREEN, ORANGE, PURPLE = "#2166ac", "#b2182b", "#1a9850", "#e08214", "#762a83"


# --------------------------------------------------------------------------- #
# loading helpers
# --------------------------------------------------------------------------- #
def load_results(tag):
    p = C.OUT_DIR / f"results_{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_preds(tag):
    out = {}
    for k in range(1, C.N_FOLDS + 1):
        p = PRED_DIR / f"{tag}_fold{k}.json"
        if p.exists():
            out[k] = json.loads(p.read_text())
    return out


def pooled(preds, split):
    """Concatenate a split across folds.  For 'test' the folds are source-disjoint,
    so the pooled set contains each of the 480 tested images exactly once."""
    y, p, extra = [], [], {"agic": [], "loe": [], "method": [], "source_id": []}
    for k in sorted(preds):
        d = preds[k][split]
        y += d["y"]
        p += d["p"]
        for kk in extra:
            extra[kk] += d.get(kk, [np.nan] * len(d["y"]))
    return np.array(y), np.array(p), extra


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(C.FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.png")


def _scatter(ax, y, p, colour, title):
    ax.scatter(y, p, s=13, alpha=0.55, c=colour, edgecolors="none")
    lo = float(min(np.min(y), np.min(p)))
    hi = float(max(np.max(y), np.max(p)))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="ideal (y = x)")
    if len(y) > 2:
        b, a = np.polyfit(y, p, 1)
        xs = np.linspace(lo - pad, hi + pad, 50)
        ax.plot(xs, b * xs + a, lw=1.4, color="k", alpha=0.75, label="least-squares fit")
    m = all_metrics(y, p)
    ax.set_xlabel("Target quality score (mean DSI)")
    ax.set_ylabel("Predicted GICQI")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.text(0.97, 0.05,
            f"n = {m['n']}\nPLCC = {m['PLCC']:.3f}\nSROCC = {m['SROCC']:.3f}\n"
            f"MAE = {m['MAE']:.3f}\nRMSE = {m['RMSE']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox=dict(fc="white", ec="0.8", alpha=0.9, boxstyle="round,pad=0.35"))
    return m


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_scatter_train_test(tag="gisrnet_tl"):
    preds = load_preds(tag)
    if not preds:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))
    for ax, split, colour, label in zip(
            axes, ("train", "val", "test"), (BLUE, ORANGE, RED),
            ("Training partition", "Validation partition", "Test partition")):
        y, p, _ = pooled(preds, split)
        _scatter(ax, y, p, colour, f"({'abc'[list(axes).index(ax)]}) {label}")
    fig.suptitle("GISR-Net: predicted GICQI versus target mean DSI "
                 "(pooled over the six source-disjoint folds)", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig_scatter_train_test")


def fig_scatter_per_fold(tag="gisrnet_tl"):
    preds = load_preds(tag)
    if not preds:
        return
    n = len(preds)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, k in zip(axes.ravel(), sorted(preds)):
        d = preds[k]["test"]
        _scatter(ax, np.array(d["y"]), np.array(d["p"]), BLUE,
                 f"Fold {k} - test partition (5 unseen source images)")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Per-fold test-set performance of GISR-Net under "
                 "source-image-disjoint cross-validation", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig_scatter_per_fold")


def fig_iqa_vs_dsi(tag="gisrnet_tl"):
    preds = load_preds(tag)
    if not preds:
        return
    y, p, ex = pooled(preds, "test")
    agic = np.array(ex["agic"], float)
    loe = np.array(ex["loe"], float)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))
    panels = [(loe, "LOE", "Lightness-order error (LOE)", GREEN),
              (agic, "AGIC", "Average gradient of illumination component (AGIC)", PURPLE),
              (p, "GICQI", "Proposed GICQI (GISR-Net)", RED)]
    for i, (ax, (v, short, long, col)) in enumerate(zip(axes, panels)):
        m = np.isfinite(v) & np.isfinite(y)
        ax.scatter(y[m], v[m], s=13, alpha=0.55, c=col, edgecolors="none")
        if m.sum() > 2:
            b, a = np.polyfit(y[m], v[m], 1)
            xs = np.linspace(y[m].min(), y[m].max(), 50)
            ax.plot(xs, b * xs + a, "k-", lw=1.4, alpha=0.8)
        ax.set_xlabel("Target quality score (mean DSI)")
        ax.set_ylabel(long)
        ax.set_title(f"({'abc'[i]}) {short} versus mean DSI", fontsize=10)
        ax.text(0.97, 0.95,
                f"|PLCC| = {abs(plcc(y[m], v[m])):.2f}\n"
                f"|SROCC| = {abs(srocc(y[m], v[m])):.2f}\n"
                f"|KROCC| = {abs(krocc(y[m], v[m])):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(fc="white", ec="0.8", alpha=0.9, boxstyle="round,pad=0.35"))
    fig.suptitle("Agreement of illumination-correction IQA metrics with segmentation "
                 f"accuracy on the pooled source-disjoint test partitions (n = {len(y)})",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "fig_iqa_vs_dsi")


def fig_correlation_bars(tag="gisrnet_tl"):
    preds = load_preds(tag)
    if not preds:
        return
    y, p, ex = pooled(preds, "test")
    series = {"LOE": np.array(ex["loe"], float),
              "AGIC": np.array(ex["agic"], float),
              "GICQI\n(proposed)": p}
    order = ["LOE", "AGIC"] + [k for k in series if k not in ("LOE", "AGIC")]
    metrics = ("PLCC", "SROCC", "KROCC")
    vals = {mm: [] for mm in metrics}
    for name in order:
        v = series[name]
        m = np.isfinite(v) & np.isfinite(y)
        vals["PLCC"].append(abs(plcc(y[m], v[m])))
        vals["SROCC"].append(abs(srocc(y[m], v[m])))
        vals["KROCC"].append(abs(krocc(y[m], v[m])))

    x = np.arange(len(order))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (mm, col) in enumerate(zip(metrics, (BLUE, ORANGE, GREEN))):
        b = ax.bar(x + (i - 1) * w, vals[mm], w, label=mm, color=col, edgecolor="white")
        ax.bar_label(b, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Absolute correlation with mean DSI")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, ncol=3)
    ax.set_title("Agreement of each IQA metric with segmentation accuracy\n"
                 "(pooled source-disjoint test partitions)", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig_correlation_bars")



def fig_learning_curves(tag="gisrnet_tl"):
    r = load_results(tag)
    if not r or not r.get("folds"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    cmap = plt.get_cmap("viridis")
    for i, f in enumerate(r["folds"]):
        h = f["history"]
        ep = [x["epoch"] for x in h]
        c = cmap(i / max(len(r["folds"]) - 1, 1))
        axes[0].plot(ep, [x["train_mse"] for x in h], color=c, lw=1.5,
                     label=f"fold {f['fold']}")
        axes[1].plot(ep, [x["val_RMSE"] for x in h], color=c, lw=1.5,
                     label=f"fold {f['fold']}")
        axes[1].scatter([f["best_epoch"]],
                        [h[f["best_epoch"] - 1]["val_RMSE"]], color=c, s=28, zorder=4)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss (MSE)")
    axes[0].set_title("(a) Training loss", fontsize=10); axes[0].set_yscale("log")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation RMSE")
    axes[1].set_title("(b) Validation RMSE (dot = selected epoch)", fontsize=10)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("GISR-Net convergence across the six source-disjoint folds", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig_learning_curves")


def fig_bland_altman(tag="gisrnet_tl"):
    preds = load_preds(tag)
    if not preds:
        return
    y, p, _ = pooled(preds, "test")
    mean = (y + p) / 2
    diff = p - y
    md, sd = diff.mean(), diff.std()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(mean, diff, s=14, alpha=0.55, c=BLUE, edgecolors="none")
    ax.axhline(md, color=RED, lw=1.4, label=f"bias = {md:+.3f}")
    ax.axhline(md + 1.96 * sd, color=RED, ls="--", lw=1,
               label=f"95% limits = {md - 1.96 * sd:+.3f}, {md + 1.96 * sd:+.3f}")
    ax.axhline(md - 1.96 * sd, color=RED, ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("Mean of GICQI and target mean DSI")
    ax.set_ylabel("GICQI - target mean DSI")
    ax.set_title("Bland-Altman agreement between GICQI and segmentation accuracy\n"
                 "(pooled source-disjoint test partitions)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_bland_altman")


def fig_residuals(tag="gisrnet_tl"):
    preds = load_preds(tag)
    if not preds:
        return
    y, p, _ = pooled(preds, "test")
    res = p - y
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].scatter(p, res, s=13, alpha=0.55, c=BLUE, edgecolors="none")
    axes[0].axhline(0, color=RED, lw=1.2)
    axes[0].set_xlabel("Predicted GICQI"); axes[0].set_ylabel("Residual")
    axes[0].set_title("(a) Residual versus prediction", fontsize=10)
    axes[1].hist(res, bins=32, color=BLUE, alpha=0.85, edgecolor="white")
    axes[1].axvline(0, color=RED, lw=1.2)
    axes[1].set_xlabel("Residual (GICQI - mean DSI)"); axes[1].set_ylabel("Count")
    axes[1].set_title(f"(b) Residual distribution "
                      f"($\\mu$ = {res.mean():+.3f}, $\\sigma$ = {res.std():.3f})",
                      fontsize=10)
    fig.tight_layout()
    _save(fig, "fig_residuals")



def fig_external_scatter():
    p = C.OUT_DIR / "results_external.json"
    if not p.exists():
        return
    ext = json.loads(p.read_text())
    sets = [(k, v) for k, v in ext.items() if v.get("models")]
    if not sets:
        return
    fig, axes = plt.subplots(1, len(sets), figsize=(5.6 * len(sets), 4.4), squeeze=False)
    for i, (name, e) in enumerate(sets):
        ax = axes[0][i]
        tag = "gisrnet_tl" if "gisrnet_tl" in e["models"] else list(e["models"])[0]
        r = e["models"][tag]
        y, pp = np.array(r["y"]), np.array(r["p_ensemble"])
        _scatter(ax, y, pp, GREEN,
                 f"({'ab'[i]}) {name}  -  {e['n_images']} images / "
                 f"{e['n_sources']} unseen sources")
    fig.suptitle("Generalisation of GISR-Net to data never seen during training",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "fig_external_scatter")


def fig_ablation():
    spec = [("gisrnet_tl", "Transfer learning\n+ augmentation\n+ linear head\n(proposed)"),
            ("gisrnet_ri", "Random\ninitialisation"),
            ("gisrnet_noaug", "No\naugmentation"),
            ("gisrnet_mlp", "MLP\nregression head"),
            ("gisrnet_sigmoid", "Sigmoid-bounded\nregression head")]
    labels, mae, rmse, sd1, sd2 = [], [], [], [], []
    for t, lab in spec:
        r = load_results(t)
        if not r or "mean_test" not in r:
            continue
        labels.append(lab)
        mae.append(r["mean_test"]["MAE"]); sd1.append(r["mean_test"]["MAE_std"])
        rmse.append(r["mean_test"]["RMSE"]); sd2.append(r["mean_test"]["RMSE_std"])
    if not labels:
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8.5, 1.9 * len(labels)), 4.3))
    b1 = ax.bar(x - 0.19, mae, 0.36, yerr=sd1, capsize=3, label="MAE",
                color=BLUE, edgecolor="white")
    b2 = ax.bar(x + 0.19, rmse, 0.36, yerr=sd2, capsize=3, label="RMSE",
                color=RED, edgecolor="white")
    ax.bar_label(b1, fmt="%.3f", fontsize=7, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Error on held-out test partitions")
    ax.set_title("Ablation study: contribution of transfer learning, augmentation "
                 "and regression-head design\n(mean $\\pm$ SD over six source-disjoint folds)",
                 fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig_ablation")


def fig_per_method_error(tag="gisrnet_tl"):
    preds = load_preds(tag)
    if not preds:
        return
    y, p, ex = pooled(preds, "test")
    methods = np.array(ex["method"])
    uniq = sorted(set(methods.tolist()))
    if len(uniq) < 2:
        return
    err = np.abs(p - y)
    data = [err[methods == m] for m in uniq]
    order = np.argsort([np.mean(d) for d in data])
    uniq = [uniq[i] for i in order]
    data = [data[i] for i in order]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    bp = axes[0].boxplot(data, patch_artist=True, widths=0.6)
    axes[0].set_xticks(np.arange(1, len(uniq) + 1))
    axes[0].set_xticklabels(uniq)
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE); patch.set_alpha(0.5)
    for med in bp["medians"]:
        med.set_color("k")
    axes[0].set_ylabel("Absolute prediction error")
    axes[0].set_title("(a) GISR-Net absolute error by illumination-correction algorithm",
                      fontsize=10)

    mt = [np.mean(y[methods == m]) for m in uniq]
    mp = [np.mean(p[methods == m]) for m in uniq]
    xx = np.arange(len(uniq))
    axes[1].bar(xx - 0.19, mt, 0.36, label="target mean DSI", color=GREEN,
                edgecolor="white")
    axes[1].bar(xx + 0.19, mp, 0.36, label="predicted GICQI", color=ORANGE,
                edgecolor="white")
    axes[1].set_xticks(xx); axes[1].set_xticklabels(uniq, rotation=45, ha="right")
    axes[1].set_ylabel("Quality score")
    axes[1].set_title("(b) Mean target and mean predicted score per algorithm", fontsize=10)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig_per_method_error")


def fig_leakage_diagram():
    folds_path = C.OUT_DIR / "folds.json"
    if not folds_path.exists():
        return
    info = json.loads(folds_path.read_text())
    blocks = []
    seen = []
    for f in info["folds"]:
        seen.append((f["test"], f["val"]))
    all_src = sorted({s for f in info["folds"] for s in f["train"] + f["val"] + f["test"]})
    pos = {s: i for i, s in enumerate(all_src)}

    fig, ax = plt.subplots(figsize=(13, 4.2))
    cols = {"train": "#c6dbef", "val": ORANGE, "test": RED}
    for r, f in enumerate(info["folds"]):
        for role in ("train", "val", "test"):
            for s in f[role]:
                ax.add_patch(Rectangle((pos[s], r), 1, 0.86,
                                       facecolor=cols[role], edgecolor="white", lw=0.4))
    ax.set_xlim(0, len(all_src)); ax.set_ylim(0, len(info["folds"]))
    ax.set_yticks(np.arange(len(info["folds"])) + 0.43)
    ax.set_yticklabels([f"Fold {i + 1}" for i in range(len(info["folds"]))])
    ax.set_xticks(np.arange(len(all_src)) + 0.5)
    ax.set_xticklabels(all_src, fontsize=6, rotation=90)
    ax.set_xlabel("Original source photograph (each contributes 16 corrected images)")
    ax.grid(False)
    handles = [Rectangle((0, 0), 1, 1, facecolor=cols[k]) for k in ("train", "val", "test")]
    ax.legend(handles, ["Training (40 sources / 640 images)",
                        "Validation (5 sources / 80 images)",
                        "Test (5 sources / 80 images)"],
              ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.set_title("Source-image-disjoint 80/10/10 partitioning used in every fold\n"
                 "(no corrected version of a held-out photograph is ever seen in training)",
                 fontsize=11, pad=10)
    fig.tight_layout()
    _save(fig, "fig_leakage_diagram")


def main():
    print(f"Writing figures to {C.FIG_DIR}")
    for fn in (fig_leakage_diagram, fig_scatter_train_test, fig_scatter_per_fold,
               fig_iqa_vs_dsi, fig_correlation_bars,
               fig_learning_curves, fig_bland_altman, fig_residuals,
               fig_external_scatter, fig_ablation,
               fig_per_method_error):
        try:
            fn()
        except Exception as e:                      # a missing run must not stop the rest
            print(f"  ! {fn.__name__} skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
