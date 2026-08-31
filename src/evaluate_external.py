"""
Generalisability evaluation on the two external test sets.

Set A - HF_Test   : 100 images, simulated uneven illumination (same database).
Set B - MED-NODE  : 80 images from an INDEPENDENT database and camera source,
                    with mean-DSI targets recomputed here from the MED-NODE
                    expert masks (see build_external_sets.py).

Each of the six fold checkpoints is evaluated separately (giving a mean +- SD
over folds) and the six predictions are also averaged into a fold ensemble.
The classical metrics AGIC and LOE are evaluated on the same images so that the
comparison is like for like.

    python3 evaluate_external.py --tags gisrnet_tl
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
from dataset import IQADataset
from metrics import all_metrics, agg
from gisrnet import build_model


def read_external(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            if k not in ("rel_path", "source_id"):
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = float("nan")
    return rows


@torch.no_grad()
def predict_set(model, rows, device, batch_size=16, workers=0):
    ds = IQADataset(rows, train=False, target_key="DSI")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    model.eval()
    ys, ps = [], []
    for x, y, _ in dl:
        ps.append(model(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def evaluate_tag(tag, rows, device, set_name):
    per_fold, preds = [], []
    for k in range(1, C.N_FOLDS + 1):
        ck = C.MODEL_DIR / f"{tag}_fold{k}.pt"
        if not ck.exists():
            continue
        blob = torch.load(ck, map_location="cpu")
        model = build_model(blob["arch"], pretrained=False,
                            head=blob.get("head", "linear")).to(device)
        model.load_state_dict(blob["state_dict"])
        y, p = predict_set(model, rows, device)
        m = all_metrics(y, p)
        m["fold"] = k
        per_fold.append(m)
        preds.append(p)
        print(f"    {tag} fold {k} on {set_name}: MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} "
              f"PLCC={m['PLCC']:.3f} SROCC={m['SROCC']:.3f}", flush=True)
        del model

    if not per_fold:
        return None
    P = np.mean(np.stack(preds), axis=0)
    ens = all_metrics(y, P)
    return {"per_fold": per_fold, "mean_over_folds": agg(per_fold),
            "fold_ensemble": ens, "y": y.tolist(), "p_ensemble": P.tolist()}


def classical_agreement(rows, key):
    y = np.array([r["DSI"] for r in rows], float)
    x = np.array([r.get(key, np.nan) for r in rows], float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return None
    r = all_metrics(y[m], x[m])
    # a classical metric is not on the DSI scale, so MAE/RMSE are meaningless
    r["MAE"] = float("nan")
    r["RMSE"] = float("nan")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["gisrnet_tl"])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    out = {}
    for set_name, csv_name in (("HF_Test", "external_hf_test.csv"),
                               ("MED-NODE", "external_mednode_test.csv")):
        path = C.OUT_DIR / csv_name
        if not path.exists():
            print(f"  ! {path} missing, run build_external_sets.py first")
            continue
        rows = read_external(path)
        print(f"\n### {set_name}: {len(rows)} images, "
              f"{len(set(r['source_id'] for r in rows))} distinct sources")
        entry = {"n_images": len(rows),
                 "n_sources": len(set(r["source_id"] for r in rows)),
                 "models": {}, "classical": {}}
        for key in ("AGIC", "LOE"):
            c = classical_agreement(rows, key)
            if c:
                entry["classical"][key] = c
                print(f"    {key}: PLCC={c['PLCC']:.3f} SROCC={c['SROCC']:.3f} "
                      f"KROCC={c['KROCC']:.3f}")
        for tag in args.tags:
            r = evaluate_tag(tag, rows, device, set_name)
            if r:
                entry["models"][tag] = r
        out[set_name] = entry

    p = C.OUT_DIR / "results_external.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWritten {p}")

    print("\n===== external generalisability summary =====")
    for sn, e in out.items():
        print(f"\n{sn} ({e['n_images']} images / {e['n_sources']} sources)")
        for tag, r in e["models"].items():
            m = r["mean_over_folds"]
            print(f"  {tag:12s} MAE={m['MAE']:.4f}+-{m['MAE_std']:.4f}  "
                  f"RMSE={m['RMSE']:.4f}  PLCC={m['PLCC']:.3f}  "
                  f"SROCC={m['SROCC']:.3f}  KROCC={m['KROCC']:.3f}")
        for k, c in e["classical"].items():
            print(f"  {k:12s} PLCC={c['PLCC']:.3f}  SROCC={c['SROCC']:.3f}  "
                  f"KROCC={c['KROCC']:.3f}")


if __name__ == "__main__":
    main()
