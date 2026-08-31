"""
* 800 illumination-corrected images derive from only 50 SOURCE photographs
  (16 correction algorithms x 50 sources).  Every split in this code base is
  made at the SOURCE-IMAGE level, never at the image level.  See `folds.py`.
* 6 folds, each with an exact 80/10/10 split of SOURCE images
  (40 train / 5 validation / 5 test sources  ->  640 / 80 / 80 images).
* The 6 test blocks are mutually disjoint.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Path resolution: the same code must run from the Linux sandbox mount and from
# the user's macOS Desktop folder.
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent          # <repo>/src
_REPO = _HERE.parent                             # <repo>

_CANDIDATE_ROOTS = [
    _REPO / "data",       # recommended: put Data.csv and the image folders here
    _REPO,                # or at the repository root
    Path.cwd(),
    Path.cwd().parent,
]


def _resolve_root() -> Path:
    """Locate the folder containing Data.csv and the image directories.

    Override with the GISRNET_ROOT environment variable, e.g.

        export GISRNET_ROOT=/path/to/GISRNET_Work        # Linux / macOS
        set GISRNET_ROOT=D:\\Research\\GISRNET_Work        # Windows
    """
    env = os.environ.get("GISRNET_ROOT")
    if env:
        return Path(env)
    for c in _CANDIDATE_ROOTS:
        if (c / "Data.csv").exists():
            return c
    raise FileNotFoundError(
        "Could not locate Data.csv.\n"
        "Place the dataset under <repo>/data/ (see data/README.md), or set the\n"
        "GISRNET_ROOT environment variable to the folder that contains Data.csv.")


ROOT = _resolve_root()

DATA_CSV = ROOT / "Data.csv"
HF_TEST_CSV = ROOT / "HF_Test.csv"

GT_DIR = ROOT / "GT"                                  # ground truth for the 50 sources
HF_TEST_DIR = ROOT / "HF_Test"
HF_GT_DIR = ROOT / "HF_GT"
MEDNODE_ENH_DIR = ROOT / "Enhanced_MedNodeTest"       # MED-NODE, illumination corrected
MEDNODE_GT_DIR = ROOT / "MED_NODE_Test_GT"
MEDNODE_SIM_DIR = ROOT / "Simulated_MED_NODE_Test"    # simulated uneven illumination

# Results directory.  Override with GISRNET_OUT when the data lives on a slow or
# read-only volume (for example Google Drive in Colab).
OUT_DIR = Path(os.environ.get("GISRNET_OUT") or (_REPO / "results"))
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
TABLE_DIR = OUT_DIR / "tables"
WEIGHTS_DIR = _REPO / "pretrained_weights"             # user-supplied ImageNet weights
# Decoded-image cache.  Keep this on fast local storage; in Colab that means
# /content rather than the mounted Drive.
CACHE_DIR = Path(os.environ.get("GISRNET_CACHE") or (OUT_DIR / "cache"))

for _d in (OUT_DIR, FIG_DIR, MODEL_DIR, TABLE_DIR, WEIGHTS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Experimental protocol
# --------------------------------------------------------------------------- #
SEED = 1337
N_SOURCES = 50
N_BLOCKS = 10          # 10 blocks x 5 sources
BLOCK_SIZE = 5
N_FOLDS = 6            # 6 rotating folds -> 6 disjoint test blocks

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.03
MOMENTUM = 0.9
LR_DROP_FACTOR = 0.1
LR_DROP_EVERY = 10
EARLY_STOP_PATIENCE = 8
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

# ImageNet normalisation (used for every backbone)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Human-readable names of the 16 illumination-correction algorithms, keyed by the
# folder prefix that appears in Data.csv.
METHOD_NAMES = {
    "Enhanced_ECCA": "ECCA",
    "Enhanced_Fusion": "FBIC",
    "Enhanced_GCCA": "GCCA",
    "Enhanced_GCIC": "GCPEI",
    "Enhanced_IBA": "IBA",
    "Enhanced_JED": "JEDM",
    "Enhanced_Mading": "SLIE",
    "Enhanced_MAGC": "MAGC",
    "Enhanced_Wang": "NPEA",
    "Enhanced_STCR": "STCR",
    "Enhanced_PCACCA": "PCACCA",
    "Enhanced_ST": "MST",
    "Enhanced_VFGLE": "VF",
    "Enhanced_WPCCA": "WPCCA",
    "Enhanced_Zheng": "SIVC",
    "Enhanced_Zhou": "LCA",
}

# Local filenames expected in ROOT/pretrained_weights for transfer learning.
PRETRAINED_FILES = {
    "googlenet": "googlenet-1378be20.pth",
}
