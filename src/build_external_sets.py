"""
Construct the two external generalisability test sets.

Set A - HF_Test (100 images, simulated uneven illumination, same database)
        Targets are supplied in HF_Test.csv; AGIC is computed here so that the
        proposed metric can be compared against the classical metrics on the
        same images.

Set B - MED-NODE (80 images, INDEPENDENT database, different camera source)
        This set directly addresses Reviewer #1 comment 5.  The mean-DSI target
        is computed here from scratch by running Otsu / minimum-error / Kapur
        segmentation against the MED-NODE expert masks.  AGIC and LOE are also
        computed (LOE using the simulated degraded image as reference).

None of the 50 University-of-Waterloo source photographs used for training
appear in either set.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from config import (ROOT, HF_TEST_CSV, HF_TEST_DIR, MEDNODE_ENH_DIR,
                    MEDNODE_GT_DIR, MEDNODE_SIM_DIR, OUT_DIR)
from iqa_classical import mean_dsi, agic, loe


def build_hf() -> list[dict]:
    rows = []
    with open(HF_TEST_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            rel = r["Images"].replace("\\", "/")
            p = ROOT / rel
            rows.append({
                "rel_path": rel,
                "source_id": Path(rel).stem,
                "DSI": float(r["Original_DSI"]),
                "AGIC": agic(p),
            })
    return rows


def build_mednode() -> list[dict]:
    rows = []
    for p in sorted(MEDNODE_ENH_DIR.glob("*.jpg")):
        sid = p.name.split("_")[0]
        gt = MEDNODE_GT_DIR / f"{sid}_GT.png"
        deg = MEDNODE_SIM_DIR / f"{sid}_OrigMEdNodeTest.jpg"
        if not gt.exists():
            print(f"  ! no ground truth for {p.name}, skipping")
            continue
        m = mean_dsi(p, gt)
        rec = {
            "rel_path": f"{MEDNODE_ENH_DIR.name}/{p.name}",
            "source_id": sid,
            **m,
            "AGIC": agic(p),
        }
        rec["LOE"] = loe(deg, p) if deg.exists() else float("nan")
        rows.append(rec)
    return rows


def _write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    print("Building HF_Test external set ...")
    hf = build_hf()
    _write_csv(hf, OUT_DIR / "external_hf_test.csv")
    print(f"  {len(hf)} images, DSI mean={np.mean([r['DSI'] for r in hf]):.4f}")

    print("Building MED-NODE external set (computing DSI from expert masks) ...")
    mn = build_mednode()
    _write_csv(mn, OUT_DIR / "external_mednode_test.csv")
    if mn:
        print(f"  {len(mn)} images, {len(set(r['source_id'] for r in mn))} distinct sources")
        for k in ("DSI_OS", "DSI_METS", "DSI_KMES", "DSI"):
            print(f"    mean {k:9s} = {np.mean([r[k] for r in mn]):.4f}")

    summary = {
        "hf_test": {"n_images": len(hf), "n_sources": len(set(r["source_id"] for r in hf)),
                    "database": "University of Waterloo (simulated uneven illumination)"},
        "mednode": {"n_images": len(mn), "n_sources": len(set(r["source_id"] for r in mn)),
                    "database": "MED-NODE (independent database / camera source)",
                    "target": "mean DSI of Otsu, minimum-error and Kapur segmentation"},
    }
    (OUT_DIR / "external_sets_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nWritten to", OUT_DIR)


if __name__ == "__main__":
    main()
