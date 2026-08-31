#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# GISR-Net revision -- run every experiment.
#
#   cd ~/Desktop/GISRNET_Work/GISRNet_PyTorch
#   bash run_all.sh
#
# On an Apple-Silicon Mac the scripts use the MPS backend automatically.
# The runner is RESUMABLE: train.py skips folds already recorded in
# results_<tag>.json, so if you stop it (Ctrl-C) and re-run, it continues.
#
# Total: 5 configurations x 6 source-disjoint folds x 30 epochs.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/../src"

echo "=== checking dependencies ==="
python3 - <<'PY'
import importlib.util, sys
missing = [m for m in ("torch", "torchvision", "numpy", "scipy",
                       "matplotlib", "pandas", "PIL", "docx")
           if importlib.util.find_spec(m) is None]
if missing:
    print("Missing packages:", ", ".join(missing))
    print("Install with:  pip3 install torch torchvision numpy scipy matplotlib pandas pillow python-docx")
    sys.exit(1)
import torch
dev = "cuda" if torch.cuda.is_available() else (
      "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
      else "cpu")
print(f"torch {torch.__version__}  device={dev}")
PY
[ $? -ne 0 ] && exit 1

RES="$(python3 -c 'import config;print(config.OUT_DIR)')"
LOG="$RES/logs"
mkdir -p "$LOG"

# BATCH: effective batch size is always 32.  ACCUM>1 splits it into smaller
# micro-batches for machines with little memory; set ACCUM=1 on a Mac with
# 16 GB or more.
ACCUM="${ACCUM:-1}"
EPOCHS="${EPOCHS:-30}"
WORKERS="${WORKERS:-2}"

run () {
  local tag="$1"; shift
  echo ""
  echo "[$(date +%H:%M:%S)] >>> $tag" | tee -a "$LOG/run_all.log"
  # --auto-lr applies the SAME learning-rate selection procedure to every
  # architecture, on the fold-1 validation partition only.  Everything else
  # (optimiser, schedule, batch size, augmentation, folds, target) is identical.
  python3 -W ignore train.py --tag "$tag" --workers "$WORKERS" --auto-lr \
          --accum "$ACCUM" --epochs "$EPOCHS" "$@" 2>&1 | tee -a "$LOG/$tag.log"
  echo "[$(date +%H:%M:%S)] <<< $tag done" | tee -a "$LOG/run_all.log"
}

echo ""
echo "=== step 0: fold definition and leakage check ==="
python3 folds.py || exit 1

echo ""
echo "=== step 1: external test sets (HF_Test + MED-NODE) ==="
python3 build_external_sets.py

echo ""
echo "=== step 2: train GISR-Net ==="
run gisrnet_tl --arch gisrnet

echo ""
echo "=== step 3: ablation study ==="
run gisrnet_ri      --arch gisrnet --no-pretrained
run gisrnet_noaug   --arch gisrnet --no-augment
run gisrnet_mlp     --arch gisrnet --head mlp
run gisrnet_sigmoid --arch gisrnet --head sigmoid

echo ""
echo "=== step 4: generalisability on the external sets ==="
python3 evaluate_external.py --tags gisrnet_tl

echo ""
echo "=== step 5: figures, tables and documents ==="
python3 make_figures.py
python3 make_tables.py
python3 make_manuscript.py

echo ""
echo "[$(date +%H:%M:%S)] ALL RUNS COMPLETE" | tee -a "$LOG/run_all.log"
echo "Results are in: $RES"
