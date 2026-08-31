"""
Dataset and augmentation.

Augmentation follows Section 2.2.2 of the manuscript: horizontal flip, vertical
flip, rotation in [-20 deg, +20 deg] and scaling with a 0.1 jitter.  It is applied
online to the training partition only; validation and test partitions are only
resized to 224x224, so no augmented copy of a held-out source image can ever be
seen during training.

Decoded images are cached in RAM at 256x256 to keep the CPU-only training loop
input-bound rather than JPEG-bound.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms as T

from config import ROOT, IMG_SIZE, MEAN, STD

CACHE_SIZE = 256


def _load_cached(rel_path: str) -> np.ndarray:
    img = Image.open(ROOT / rel_path).convert("RGB").resize(
        (CACHE_SIZE, CACHE_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


class ImageCache:
    """Shared decoded-image cache keyed by relative path.

    Optionally backed by an on-disk memory-mapped array so that repeated runs
    (and the resumable runner) do not pay the JPEG-decoding cost again.
    """

    def __init__(self):
        self._d = {}
        self._mm = None
        self._idx = {}

    # -- in-memory -------------------------------------------------------- #
    def get(self, rel_path: str) -> np.ndarray:
        j = self._idx.get(rel_path)
        if j is not None and self._mm is not None:
            return self._mm[j]
        a = self._d.get(rel_path)
        if a is None:
            a = _load_cached(rel_path)
            self._d[rel_path] = a
        return a

    def preload(self, rel_paths: Sequence[str], verbose: bool = True) -> None:
        for i, p in enumerate(rel_paths):
            self.get(p)
            if verbose and (i + 1) % 200 == 0:
                print(f"    cached {i + 1}/{len(rel_paths)}", flush=True)

    # -- on-disk ---------------------------------------------------------- #
    def attach_disk_cache(self, rel_paths: Sequence[str], cache_dir: Path,
                          name: str = "image_cache_256", verbose: bool = True) -> None:
        import json
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        arr_p = cache_dir / f"{name}.npy"
        idx_p = cache_dir / f"{name}.json"
        paths = list(rel_paths)

        if arr_p.exists() and idx_p.exists():
            idx = json.loads(idx_p.read_text())
            if idx.get("paths") == paths:
                self._mm = np.load(arr_p, mmap_mode="r")
                self._idx = {p: i for i, p in enumerate(paths)}
                if verbose:
                    print(f"    attached disk cache {arr_p.name} "
                          f"({self._mm.shape[0]} images)", flush=True)
                return

        if verbose:
            print(f"    building disk cache {arr_p.name} ...", flush=True)
        mm = np.lib.format.open_memmap(
            arr_p, mode="w+", dtype=np.uint8,
            shape=(len(paths), CACHE_SIZE, CACHE_SIZE, 3))
        for i, p in enumerate(paths):
            mm[i] = _load_cached(p)
            if verbose and (i + 1) % 200 == 0:
                print(f"      {i + 1}/{len(paths)}", flush=True)
        mm.flush()
        del mm
        idx_p.write_text(json.dumps({"paths": paths}))
        self._mm = np.load(arr_p, mmap_mode="r")
        self._idx = {p: i for i, p in enumerate(paths)}


GLOBAL_CACHE = ImageCache()


def build_transforms(train: bool, augment: bool = True):
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if train and augment:
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomAffine(degrees=20, scale=(0.9, 1.1),
                           interpolation=T.InterpolationMode.BILINEAR),
            T.Resize((IMG_SIZE, IMG_SIZE)),
            *norm,
        ])
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), *norm])


class IQADataset(Dataset):
    """Returns (image_tensor, target, index)."""

    def __init__(self, records, train: bool, augment: bool = True,
                 cache: ImageCache | None = None, target_key: str = "dsi"):
        self.records = list(records)
        self.tf = build_transforms(train, augment)
        self.cache = cache or GLOBAL_CACHE
        self.target_key = target_key

    def __len__(self) -> int:
        return len(self.records)

    def _target(self, r) -> float:
        if isinstance(r, dict):
            return float(r[self.target_key.upper()] if self.target_key.upper() in r
                         else r[self.target_key])
        return float(getattr(r, self.target_key))

    def _path(self, r) -> str:
        return r["rel_path"] if isinstance(r, dict) else r.rel_path

    def __getitem__(self, i):
        r = self.records[i]
        arr = np.asarray(self.cache.get(self._path(r)))
        img = Image.fromarray(arr)
        return self.tf(img), torch.tensor(self._target(r), dtype=torch.float32), i
