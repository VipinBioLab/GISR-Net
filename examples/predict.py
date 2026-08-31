"""
Predict the GICQI quality score for one or more illumination-corrected
dermatological macro-photographs using a trained GISR-Net checkpoint.

Examples
--------
    # single image, single fold checkpoint
    python examples/predict.py --image path/to/corrected.jpg \
        --checkpoint results/models/gisrnet_tl_fold1.pt

    # a whole folder, averaging the six fold checkpoints (recommended)
    python examples/predict.py --dir path/to/images \
        --checkpoint results/models/gisrnet_tl_fold*.pt --csv scores.csv

The GICQI is on the same [0, 1] scale as the Dice similarity index: a higher
value predicts that intensity-based segmentation of the lesion will be more
accurate on that image.  See the caveats in the README before using an
individual score as a substitute for measuring Dice directly.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torchvision.transforms as T          # noqa: E402
from gisrnet import build_model             # noqa: E402
import config as C                          # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(patterns, device):
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)) or ([pat] if Path(pat).exists() else []))
    if not paths:
        raise FileNotFoundError(f"no checkpoint matched: {patterns}")

    models = []
    for p in paths:
        blob = torch.load(p, map_location="cpu")
        m = build_model(blob.get("arch", "gisrnet"), pretrained=False,
                        head=blob.get("head", "linear"))
        m.load_state_dict(blob["state_dict"])
        m.eval().to(device)
        models.append((Path(p).name, m))
    return models


def collect_images(args) -> list[Path]:
    if args.image:
        return [Path(i) for i in args.image]
    d = Path(args.dir)
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in EXTS)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Predict GICQI with GISR-Net")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", nargs="+", help="one or more image files")
    g.add_argument("--dir", help="a folder of images (searched recursively)")
    ap.add_argument("--checkpoint", nargs="+", required=True,
                    help="checkpoint path(s); globs are expanded and averaged")
    ap.add_argument("--csv", help="write the scores to this CSV file")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    device = torch.device(pick_device(args.device))
    models = load_models(args.checkpoint, device)
    images = collect_images(args)
    if not images:
        print("no images found")
        return
    print(f"{len(images)} image(s), {len(models)} checkpoint(s), device={device}\n")

    tf = T.Compose([T.Resize((C.IMG_SIZE, C.IMG_SIZE)),
                    T.ToTensor(), T.Normalize(C.MEAN, C.STD)])

    rows = []
    for i in range(0, len(images), args.batch_size):
        chunk = images[i:i + args.batch_size]
        batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk]).to(device)
        preds = np.stack([m(batch).cpu().numpy() for _, m in models])   # (folds, B)
        for j, p in enumerate(chunk):
            per_fold = preds[:, j]
            rows.append({
                "image": str(p),
                "GICQI": float(per_fold.mean()),
                "GICQI_std": float(per_fold.std()) if len(models) > 1 else 0.0,
            })

    width = min(max((len(Path(r["image"]).name) for r in rows), default=20), 55)
    print(f"{'image':<{width}}  {'GICQI':>7}  {'SD':>6}")
    print("-" * (width + 18))
    for r in rows:
        name = Path(r["image"]).name
        print(f"{name[:width]:<{width}}  {r['GICQI']:7.4f}  {r['GICQI_std']:6.4f}")

    vals = np.array([r["GICQI"] for r in rows])
    print("-" * (width + 18))
    print(f"{'mean':<{width}}  {vals.mean():7.4f}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["image", "GICQI", "GICQI_std"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten {args.csv}")


if __name__ == "__main__":
    main()
