"""
Training driver for GISR-Net and the two NR-IQA baselines.

Every experiment uses the SAME six source-image-disjoint 80/10/10 folds, the
same optimiser (SGDM, lr 0.03, momentum 0.9, step decay 0.1 every 10 epochs),
the same batch size (32), the same augmentation and the same target (mean DSI
of Otsu / minimum-error / Kapur segmentation).

Examples
--------
    python3 train.py --arch gisrnet --tag gisrnet_tl
    python3 train.py --arch gisrnet --tag gisrnet_ri   --no-pretrained
    python3 train.py --arch gisrnet --tag gisrnet_noaug --no-augment
    python3 train.py --arch gisrnet --tag gisrnet_mlp  --head mlp
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as C
from folds import load_samples, make_folds, split_samples, assert_no_leakage
from dataset import IQADataset, GLOBAL_CACHE
from metrics import all_metrics, agg
from gisrnet import build_model, count_parameters

PRED_DIR = C.OUT_DIR / "predictions"
LOG_DIR = C.OUT_DIR / "logs"
for d in (PRED_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def pick_device(pref: str = "auto") -> str:
    """Prefer CUDA, then Apple-Silicon MPS, then CPU."""
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(s: int) -> None:
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y, _ in loader:
        p = model(x.to(device))
        ys.append(y.numpy())
        ps.append(p.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def measure_inference_time(model, device, n: int = 20) -> float:
    model.eval()
    x = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        t0 = time.time()
        for _ in range(n):
            model(x)
    return (time.time() - t0) / n


def probe_lr(args, fold, samples, device) -> dict:
    """Select the learning rate for this architecture on the VALIDATION partition.

    The optimiser, schedule shape, batch size, augmentation, folds and target are
    held fixed; only the initial learning rate is selected, using a short probe on
    the training and validation partitions of the FIRST fold.  The test partitions
    are never involved in this choice.
    """
    grid = [float(x) for x in args.lr_grid.split(",")]
    sp = split_samples(samples, fold)
    micro = args.batch_size // args.accum
    dl = lambda d, sh: DataLoader(d, batch_size=micro, shuffle=sh,
                                  num_workers=args.workers, drop_last=False)
    dl_tr = dl(IQADataset(sp["train"], train=True, augment=args.augment), True)
    dl_va = dl(IQADataset(sp["val"], train=False), False)
    crit = lambda pred, target: 0.5 * nn.functional.mse_loss(pred, target)

    print(f"\n=== {args.tag} | learning-rate probe on fold 1 "
          f"({args.lr_probe_epochs} epochs per candidate, validation partition only) ===")
    scores = {}
    for lr in grid:
        set_seed(C.SEED)
        model = build_model(args.arch, pretrained=args.pretrained, head=args.head).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=C.MOMENTUM,
                              weight_decay=C.WEIGHT_DECAY)
        best = np.inf
        for _ in range(args.lr_probe_epochs):
            model.train()
            opt.zero_grad(set_to_none=True)
            for step, (x, y, _) in enumerate(dl_tr):
                x, y = x.to(device), y.to(device)
                (crit(model(x), y) / args.accum).backward()
                if (step + 1) % args.accum == 0 or (step + 1) == len(dl_tr):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            yv, pv = predict(model, dl_va, device)
            r = all_metrics(yv, pv)["RMSE"]
            if np.isfinite(r):
                best = min(best, r)
        scores[lr] = float(best)
        print(f"    lr = {lr:<8g} best validation RMSE = "
              f"{'diverged' if not np.isfinite(best) else f'{best:.4f}'}", flush=True)
        del model, opt

    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    chosen = min(finite, key=finite.get) if finite else grid[-1]
    print(f"    -> selected lr = {chosen}")
    return {"selected": chosen, "scores": scores,
            "probe_epochs": args.lr_probe_epochs, "grid": grid}


def run_fold(args, fold_idx, fold, samples, device) -> dict:
    sp = split_samples(samples, fold)
    print(f"\n=== {args.tag} | fold {fold_idx + 1}/{C.N_FOLDS} ===")
    print(f"    sources  train/val/test = {len(fold['train'])}/{len(fold['val'])}/{len(fold['test'])}")
    print(f"    images   train/val/test = {len(sp['train'])}/{len(sp['val'])}/{len(sp['test'])}")

    ds_tr = IQADataset(sp["train"], train=True, augment=args.augment)
    ds_tr_eval = IQADataset(sp["train"], train=False)
    ds_va = IQADataset(sp["val"], train=False)
    ds_te = IQADataset(sp["test"], train=False)

    micro = args.batch_size // args.accum
    dl = lambda d, sh: DataLoader(d, batch_size=micro, shuffle=sh,
                                  num_workers=args.workers, drop_last=False)
    dl_tr, dl_tr_eval = dl(ds_tr, True), dl(ds_tr_eval, False)
    dl_va, dl_te = dl(ds_va, False), dl(ds_te, False)

    set_seed(C.SEED + fold_idx)
    model = build_model(args.arch, pretrained=args.pretrained, head=args.head).to(device)
    # Eq. (11) of the manuscript: loss = 1/2 * sum (D_i - G_i)^2, averaged over the
    # mini-batch.  This is the half-MSE used by MATLAB's regression layer; keeping
    # the factor 1/2 makes the PyTorch gradients match the original implementation
    # at the same learning rate.
    crit = lambda pred, target: 0.5 * nn.functional.mse_loss(pred, target)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=C.MOMENTUM,
                          weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=C.LR_DROP_EVERY,
                                            gamma=C.LR_DROP_FACTOR)

    best = {"val_rmse": np.inf, "epoch": -1, "state": None}
    history = []
    bad = 0
    for ep in range(args.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, (x, y, _) in enumerate(dl_tr):
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            (loss / args.accum).backward()
            if (step + 1) % args.accum == 0 or (step + 1) == len(dl_tr):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            tot += loss.item()
            nb += 1
        sched.step()

        yv, pv = predict(model, dl_va, device)
        mv = all_metrics(yv, pv)
        history.append({"epoch": ep + 1, "train_mse": tot / max(nb, 1),
                        "lr": opt.param_groups[0]["lr"], **{f"val_{k}": v for k, v in mv.items()}})
        print(f"    ep {ep + 1:02d}/{args.epochs}  train_mse={tot / max(nb, 1):.5f}  "
              f"val MAE={mv['MAE']:.4f} RMSE={mv['RMSE']:.4f} PLCC={mv['PLCC']:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

        if mv["RMSE"] < best["val_rmse"] - 1e-5:
            best = {"val_rmse": mv["RMSE"], "epoch": ep + 1,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"    early stopping at epoch {ep + 1} "
                      f"(best epoch {best['epoch']})", flush=True)
                break

    model.load_state_dict(best["state"])
    ckpt = C.MODEL_DIR / f"{args.tag}_fold{fold_idx + 1}.pt"
    torch.save({"arch": args.arch, "head": args.head, "state_dict": best["state"],
                "fold": fold_idx + 1, "fold_sources": fold,
                "best_epoch": best["epoch"]}, ckpt)

    out = {"fold": fold_idx + 1, "best_epoch": best["epoch"], "history": history,
           "sources": {k: sorted(v) for k, v in fold.items()},
           "n_images": {k: len(v) for k, v in sp.items()},
           "checkpoint": str(ckpt)}

    preds = {}
    for split, loader, recs in (("train", dl_tr_eval, sp["train"]),
                                ("val", dl_va, sp["val"]),
                                ("test", dl_te, sp["test"])):
        y, p = predict(model, loader, device)
        out[split] = all_metrics(y, p)
        preds[split] = {
            "y": y.tolist(), "p": p.tolist(),
            "rel_path": [r.rel_path for r in recs],
            "source_id": [r.source_id for r in recs],
            "method": [r.method for r in recs],
            "agic": [r.agic for r in recs],
            "loe": [r.loe for r in recs],
        }
        print(f"    {split:5s}: MAE={out[split]['MAE']:.4f} RMSE={out[split]['RMSE']:.4f} "
              f"PLCC={out[split]['PLCC']:.3f} SROCC={out[split]['SROCC']:.3f} "
              f"KROCC={out[split]['KROCC']:.3f}")

    (PRED_DIR / f"{args.tag}_fold{fold_idx + 1}.json").write_text(json.dumps(preds))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gisrnet")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--head", default="linear", choices=["linear", "mlp", "sigmoid"])
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE,
                    help="EFFECTIVE batch size; kept at 32 for every model")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient-accumulation steps; micro-batch = batch_size/accum. "
                         "Used only to fit memory-heavy backbones, the effective "
                         "batch size is unchanged.")
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--auto-lr", action="store_true",
                    help="select the initial learning rate per architecture using a "
                         "short probe on the fold-1 validation partition")
    ap.add_argument("--lr-grid", default="0.03,0.01,0.003,0.001")
    ap.add_argument("--lr-probe-epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=C.EARLY_STOP_PATIENCE)
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--folds", default="1-6", help="e.g. '1-6' or '1,3,5'")
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--threads", type=int, default=0,
                    help="0 = let PyTorch choose")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.set_defaults(pretrained=True, augment=True)
    args = ap.parse_args()
    args.tag = args.tag or args.arch

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(pick_device(args.device))
    print(f"device: {device}")

    samples = load_samples()
    folds = make_folds(samples)
    assert_no_leakage(folds, samples)
    print(f"Loaded {len(samples)} images from "
          f"{len(set(s.source_id for s in samples))} source photographs")
    print("Leakage assertions passed (source-disjoint 80/10/10, 6 folds)")

    print("Caching images ...", flush=True)
    GLOBAL_CACHE.attach_disk_cache([s.rel_path for s in samples], C.CACHE_DIR)

    if "-" in args.folds:
        a, b = args.folds.split("-")
        want = list(range(int(a) - 1, int(b)))
    else:
        want = [int(x) - 1 for x in args.folds.split(",")]

    results_path = C.OUT_DIR / f"results_{args.tag}.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {"folds": []}
    done = {f["fold"] for f in results["folds"]}

    if args.auto_lr:
        if "lr_probe" in results.get("model", {}):
            args.lr = results["model"]["lr_probe"]["selected"]
            print(f"reusing previously selected lr = {args.lr}")
            lr_probe = results["model"]["lr_probe"]
        else:
            lr_probe = probe_lr(args, folds[0], samples, device)
            args.lr = lr_probe["selected"]
    else:
        lr_probe = None

    probe = build_model(args.arch, pretrained=False, head=args.head)
    results["model"] = {
        "arch": args.arch, "tag": args.tag, "head": args.head,
        "params_millions": round(count_parameters(probe), 2),
        "pretrained": args.pretrained, "augment": args.augment,
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "inference_time_s_per_image": round(measure_inference_time(probe.to(device), device), 4),
    }
    if lr_probe:
        results["model"]["lr_probe"] = lr_probe
    del probe

    for k in want:
        if (k + 1) in done:
            print(f"fold {k + 1} already complete, skipping")
            continue
        r = run_fold(args, k, folds[k], samples, device)
        results["folds"].append(r)
        results["folds"].sort(key=lambda x: x["fold"])
        for split in ("train", "val", "test"):
            results[f"mean_{split}"] = agg([f[split] for f in results["folds"]])
        results_path.write_text(json.dumps(results, indent=2))
        print(f"  -> written {results_path}")

    print(f"\n===== {args.tag}: mean over {len(results['folds'])} folds =====")
    for split in ("train", "val", "test"):
        m = results.get(f"mean_{split}")
        if m:
            print(f"  {split:5s}  MAE={m['MAE']:.4f}+-{m['MAE_std']:.4f}  "
                  f"RMSE={m['RMSE']:.4f}+-{m['RMSE_std']:.4f}  "
                  f"PLCC={m['PLCC']:.3f}  SROCC={m['SROCC']:.3f}  KROCC={m['KROCC']:.3f}")


if __name__ == "__main__":
    main()
