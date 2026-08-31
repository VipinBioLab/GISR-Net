"""
Source-image-disjoint fold construction.

This module is the direct answer to Reviewer #1 comment 2 and Reviewer #2
comment 1: *all* corrected versions (and all augmentations) of a given original
photograph are guaranteed to live in exactly one of {train, val, test} within a
fold.

Protocol
--------
The 50 original photographs are shuffled once with a fixed seed and partitioned
into 10 disjoint blocks of 5 sources.  Fold k (k = 0..5) then uses

    test  = block[k]                       ->  5 sources  ->  80 images  (10 %)
    val   = block[(k + 6) % 10]            ->  5 sources  ->  80 images  (10 %)
    train = the remaining 8 blocks         -> 40 sources  -> 640 images  (80 %)

Because k runs over 0..5 the six test blocks are mutually disjoint, and because
(k + 6) % 10 for k = 0..5 gives 6,7,8,9,0,1 the validation block is never equal
to the test block of the same fold.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict

from config import DATA_CSV, SEED, N_BLOCKS, BLOCK_SIZE, N_FOLDS, OUT_DIR, METHOD_NAMES


# --------------------------------------------------------------------------- #
# Record + loading
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    rel_path: str      # POSIX-normalised path relative to ROOT
    source_id: str     # e.g. "01" -- the ORIGINAL photograph this derives from
    method: str        # e.g. "ECCA"
    dsi: float         # target: mean DSI of Otsu / MET / Kapur
    agic: float
    loe: float


def _norm(p: str) -> str:
    return p.replace("\\", "/").strip()


def source_id_of(rel_path: str) -> str:
    """Extract the original-photograph identifier from a corrected-image path.

    'Enhanced_ECCA/01_Orig_ECCA.jpg' -> '01'
    """
    base = _norm(rel_path).split("/")[-1]
    return base.split("_")[0]


def load_samples(csv_path: Path = DATA_CSV) -> List[Sample]:
    samples: List[Sample] = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            rel = _norm(row["Name"])
            folder = rel.split("/")[0]
            samples.append(
                Sample(
                    rel_path=rel,
                    source_id=source_id_of(rel),
                    method=METHOD_NAMES.get(folder, folder),
                    dsi=float(row["DSI"]),
                    agic=float(row["AGIC"]),
                    loe=float(row["LOE"]),
                )
            )
    return samples


# --------------------------------------------------------------------------- #
# Fold construction
# --------------------------------------------------------------------------- #
def build_blocks(source_ids: List[str], seed: int = SEED) -> List[List[str]]:
    ids = sorted(set(source_ids))
    if len(ids) != N_BLOCKS * BLOCK_SIZE:
        raise ValueError(
            f"Expected {N_BLOCKS * BLOCK_SIZE} unique source images, found {len(ids)}"
        )
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    return [shuffled[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE] for i in range(N_BLOCKS)]


def make_folds(samples: List[Sample], seed: int = SEED) -> List[Dict[str, List[str]]]:
    """Return a list of {'train': [...], 'val': [...], 'test': [...]} SOURCE ids."""
    blocks = build_blocks([s.source_id for s in samples], seed)
    folds = []
    for k in range(N_FOLDS):
        test_b = k
        val_b = (k + 6) % N_BLOCKS
        assert test_b != val_b
        test_src = blocks[test_b]
        val_src = blocks[val_b]
        train_src = [s for i, b in enumerate(blocks) if i not in (test_b, val_b) for s in b]
        folds.append({"train": train_src, "val": val_src, "test": test_src})
    return folds


def split_samples(samples: List[Sample], fold: Dict[str, List[str]]):
    idx = {s: set(v) for s, v in fold.items()}
    out = {k: [x for x in samples if x.source_id in idx[k]] for k in ("train", "val", "test")}
    return out


# --------------------------------------------------------------------------- #
# Leakage assertions (Reviewer #1.2, Reviewer #2.1)
# --------------------------------------------------------------------------- #
def assert_no_leakage(folds, samples) -> None:
    all_src = set(s.source_id for s in samples)
    seen_test = set()
    for k, f in enumerate(folds):
        tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
        assert not (tr & va), f"fold {k}: train/val source overlap"
        assert not (tr & te), f"fold {k}: train/test source overlap"
        assert not (va & te), f"fold {k}: val/test source overlap"
        assert tr | va | te == all_src, f"fold {k}: sources missing from the partition"
        assert len(tr) == 40 and len(va) == 5 and len(te) == 5, f"fold {k}: wrong split sizes"
        assert not (seen_test & te), f"fold {k}: test block overlaps an earlier fold"
        seen_test |= te

        sp = split_samples(samples, f)
        assert len(sp["train"]) == 640 and len(sp["val"]) == 80 and len(sp["test"]) == 80, (
            f"fold {k}: image counts "
            f"{len(sp['train'])}/{len(sp['val'])}/{len(sp['test'])}"
        )
        # image-level disjointness (belt and braces)
        p_tr = set(x.rel_path for x in sp["train"])
        p_va = set(x.rel_path for x in sp["val"])
        p_te = set(x.rel_path for x in sp["test"])
        assert not (p_tr & p_va) and not (p_tr & p_te) and not (p_va & p_te)


def main() -> None:
    samples = load_samples()
    folds = make_folds(samples)
    assert_no_leakage(folds, samples)

    sources = sorted(set(s.source_id for s in samples))
    summary = {
        "n_images": len(samples),
        "n_source_images": len(sources),
        "n_correction_methods": len(set(s.method for s in samples)),
        "protocol": "6 rotating source-disjoint folds, 80/10/10 at SOURCE level",
        "per_fold_sources": {"train": 40, "val": 5, "test": 5},
        "per_fold_images": {"train": 640, "val": 80, "test": 80},
        "distinct_test_sources_across_folds": 30,
        "distinct_test_images_across_folds": 480,
        "folds": [
            {"fold": k + 1, **{kk: sorted(vv) for kk, vv in f.items()}}
            for k, f in enumerate(folds)
        ],
    }
    out = OUT_DIR / "folds.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=2))
    print(f"\nLeakage assertions passed. Fold definition written to {out}")


if __name__ == "__main__":
    main()
