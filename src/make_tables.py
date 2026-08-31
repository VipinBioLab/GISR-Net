"""
Assemble every table needed by the revised manuscript into
GISRNet_Results/tables/tables.json (and matching .csv files).

Tables produced
---------------
T1  Dataset and split composition          (Reviewer #1.4, Reviewer #2.2)
T2  Per-fold source and image counts       (Reviewer #2.2)
T4  Baseline CNN comparison                (values carried over from the
                                            original manuscript, clearly marked)
T5  Agreement of IQA metrics with DSI      (revised Table 5)
T6  External generalisability              (Reviewer #1.5)
T7  Ablation study                         (Reviewer #1.7)
T8  Per-fold test results for GISR-Net
"""

from __future__ import annotations

import csv
import json

import numpy as np

import config as C
from metrics import all_metrics, plcc, srocc, krocc
from make_figures import load_results, load_preds, pooled

# --------------------------------------------------------------------------- #
# Values reported in the ORIGINAL manuscript (Table 4).  These are carried over
# verbatim, as instructed, and are NOT recomputed here.  They were obtained
# under the original image-level split and are reported as such.
# --------------------------------------------------------------------------- #
ORIGINAL_CNN_TABLE = [
    # network, depth, params (M), MAE, RMSE, inference time (s/image)
    ("AlexNet [19]",          "8",   56.80, 0.11, 0.16, 2.19),
    ("Inception-v3 [21]",     "48",  21.80, 0.08, 0.15, 3.34),
    ("ResNet-50 [23]",        "50",  25.60, 0.07, 0.14, 2.77),
    ("VGG-16 [22]",           "16", 138.00, 0.16, 0.20, 4.29),
    ("DenseNet [20]",         "201", 20.00, 0.12, 0.14, 4.52),
    ("SqueezeNet [24]",       "18",   3.20, 0.25, 0.29, 2.07),
    ("MobileNetV2 [25]",      "53",   3.50, 0.14, 0.18, 2.47),
    ("NASNet-Mobile [26]",    "*",    5.30, 0.12, 0.16, 3.10),
    ("EfficientNet-b0 [26]",  "82",   5.30, 0.12, 0.16, 3.36),
    ("XceptionNet [27]",      "71",  22.90, 0.08, 0.13, 2.80),
    ("GISR-Net",              "22",   5.90, 0.06, 0.10, 2.24),
]

PRETTY = {"gisrnet_tl": "GISR-Net (proposed)",
          "gisrnet_ri": "GISR-Net, random initialisation",
          "gisrnet_noaug": "GISR-Net, no augmentation",
          "gisrnet_mlp": "GISR-Net, MLP regression head",
          "gisrnet_sigmoid": "GISR-Net, sigmoid-bounded head"}


def f(x, nd=3):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def write_csv(name, header, rows):
    p = C.TABLE_DIR / f"{name}.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {p.name}")


# --------------------------------------------------------------------------- #
def t1_dataset():
    folds = json.loads((C.OUT_DIR / "folds.json").read_text())
    ext = {}
    p = C.OUT_DIR / "external_sets_summary.json"
    if p.exists():
        ext = json.loads(p.read_text())
    rows = [
        ["Original source photographs (Univ. of Waterloo)", "50", "-"],
        ["Illumination-correction algorithms applied", "16", "-"],
        ["Illumination-corrected images in the development set", "800", "50 sources"],
        ["Training partition, per fold", "640 images", "40 sources (80%)"],
        ["Validation partition, per fold", "80 images", "5 sources (10%)"],
        ["Test partition, per fold", "80 images", "5 sources (10%)"],
        ["Distinct images tested across the 6 folds", "480 images", "30 sources"],
        ["External set A - HF_Test", f"{ext.get('hf_test', {}).get('n_images', 100)} images",
         "simulated uneven illumination"],
        ["External set B - MED-NODE", f"{ext.get('mednode', {}).get('n_images', 80)} images",
         f"{ext.get('mednode', {}).get('n_sources', 80)} independent sources"],
    ]
    write_csv("T1_dataset_composition", ["Item", "Count", "Source images"], rows)
    return {"header": ["Item", "Count", "Source images"], "rows": rows}


def t2_folds():
    info = json.loads((C.OUT_DIR / "folds.json").read_text())
    rows = []
    for fd in info["folds"]:
        rows.append([
            f"Fold {fd['fold']}",
            f"{len(fd['train'])} ({len(fd['train']) * 16} images)",
            f"{len(fd['val'])} ({len(fd['val']) * 16} images)",
            f"{len(fd['test'])} ({len(fd['test']) * 16} images)",
            ", ".join(fd["test"]),
        ])
    header = ["Fold", "Training sources", "Validation sources", "Test sources",
              "Identifiers of the held-out source photographs"]
    write_csv("T2_fold_composition", header, rows)
    return {"header": header, "rows": rows}



def t4_original_cnns():
    header = ["Network", "Depth", "Parameters (M)", "MAE", "RMSE",
              "Inference time per image (s)"]
    rows = [[n, d, f"{p:.2f}", f"{a:.2f}", f"{b:.2f}", f"{t:.2f}"]
            for n, d, p, a, b, t in ORIGINAL_CNN_TABLE]
    write_csv("T4_original_cnn_comparison", header, rows)
    return {"header": header, "rows": rows,
            "note": "Values reproduced from Table 4 of the original submission; "
                    "obtained under the original image-level split."}


def t5_iqa_agreement():
    preds = load_preds("gisrnet_tl")
    if not preds:
        return None
    y, p, ex = pooled(preds, "test")
    series = [("LOE [14]", np.array(ex["loe"], float)),
              ("AGIC [15]", np.array(ex["agic"], float))]
    series.append(("GICQI (proposed)", p))

    header = ["IQA metric", "PLCC", "SROCC", "KROCC"]
    rows = []
    for name, v in series:
        m = np.isfinite(v) & np.isfinite(y)
        rows.append([name, f(abs(plcc(y[m], v[m])), 2),
                     f(abs(srocc(y[m], v[m])), 2), f(abs(krocc(y[m], v[m])), 2)])
    write_csv("T5_iqa_agreement", header, rows)
    return {"header": header, "rows": rows,
            "note": f"Computed on the {len(y)} images of the pooled, "
                    "source-disjoint test partitions."}


def t6_external():
    p = C.OUT_DIR / "results_external.json"
    if not p.exists():
        return None
    ext = json.loads(p.read_text())
    header = ["Evaluation set", "Images", "Source images", "Model / metric",
              "MAE", "RMSE", "PLCC", "SROCC", "KROCC"]
    rows = []
    dev = load_results("gisrnet_tl")
    if dev and "mean_test" in dev:
        m = dev["mean_test"]
        rows.append(["Development set (held-out folds)", "480", "30",
                     "GISR-Net", f(m["MAE"]), f(m["RMSE"]),
                     f(m["PLCC"], 2), f(m["SROCC"], 2), f(m["KROCC"], 2)])
    for sn, e in ext.items():
        for tag, r in e["models"].items():
            m = r["mean_over_folds"]
            rows.append([sn, str(e["n_images"]), str(e["n_sources"]), PRETTY.get(tag, tag),
                         f(m["MAE"]), f(m["RMSE"]), f(m["PLCC"], 2),
                         f(m["SROCC"], 2), f(m["KROCC"], 2)])
        for k, c in e["classical"].items():
            rows.append([sn, str(e["n_images"]), str(e["n_sources"]), k,
                         "-", "-", f(c["PLCC"], 2), f(c["SROCC"], 2), f(c["KROCC"], 2)])
    write_csv("T6_generalisability", header, rows)
    return {"header": header, "rows": rows}


def t7_ablation():
    spec = [("gisrnet_tl", "Transfer learning, augmentation, linear head (proposed)"),
            ("gisrnet_ri", "Random initialisation (no transfer learning)"),
            ("gisrnet_noaug", "No data augmentation"),
            ("gisrnet_mlp", "MLP regression head"),
            ("gisrnet_sigmoid", "Sigmoid-bounded regression head")]
    header = ["Configuration", "Test MAE", "Test RMSE", "Test PLCC", "Test SROCC"]
    rows = []
    for tag, label in spec:
        r = load_results(tag)
        if not r or "mean_test" not in r:
            continue
        m = r["mean_test"]
        rows.append([label, f"{f(m['MAE'])} +- {f(m['MAE_std'])}",
                     f"{f(m['RMSE'])} +- {f(m['RMSE_std'])}",
                     f(m["PLCC"], 2), f(m["SROCC"], 2)])
    write_csv("T7_ablation", header, rows)
    return {"header": header, "rows": rows}


def t8_per_fold():
    r = load_results("gisrnet_tl")
    if not r:
        return None
    header = ["Fold", "Train images", "Val images", "Test images",
              "Best epoch", "Test MAE", "Test RMSE", "Test PLCC", "Test SROCC", "Test KROCC"]
    rows = []
    for fd in r["folds"]:
        t = fd["test"]
        rows.append([str(fd["fold"]), str(fd["n_images"]["train"]),
                     str(fd["n_images"]["val"]), str(fd["n_images"]["test"]),
                     str(fd["best_epoch"]), f(t["MAE"]), f(t["RMSE"]),
                     f(t["PLCC"], 2), f(t["SROCC"], 2), f(t["KROCC"], 2)])
    if "mean_test" in r:
        m = r["mean_test"]
        rows.append(["Mean", "640", "80", "80", "-", f(m["MAE"]), f(m["RMSE"]),
                     f(m["PLCC"], 2), f(m["SROCC"], 2), f(m["KROCC"], 2)])
        rows.append(["SD", "-", "-", "-", "-", f(m["MAE_std"]), f(m["RMSE_std"]),
                     "-", "-", "-"])
    write_csv("T8_per_fold", header, rows)
    return {"header": header, "rows": rows}


def main():
    print(f"Writing tables to {C.TABLE_DIR}")
    out = {}
    for key, fn in (("T1_dataset", t1_dataset), ("T2_folds", t2_folds),
                    ("T4_original_cnns", t4_original_cnns),
                    ("T5_iqa_agreement", t5_iqa_agreement),
                    ("T6_generalisability", t6_external),
                    ("T7_ablation", t7_ablation), ("T8_per_fold", t8_per_fold)):
        try:
            r = fn()
            if r:
                out[key] = r
        except Exception as e:
            print(f"  ! {key} skipped: {type(e).__name__}: {e}")
    (C.TABLE_DIR / "tables.json").write_text(json.dumps(out, indent=2))
    print(f"  wrote tables.json ({len(out)} tables)")


if __name__ == "__main__":
    main()
