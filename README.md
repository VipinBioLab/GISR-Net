# GISR-Net

### An image-to-scalar regressor for quality assessment of illumination correction in dermatological macro-photographs

Vipin Venugopal<sup>a</sup>, Malaya Kumar Nath<sup>b</sup>, Justin Joseph<sup>c</sup>, V. R. Simi<sup>d</sup>, Anoop B. N.<sup>e,∗</sup>

<sup>a</sup> School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore-641112, India
<sup>b</sup> Department of Electronics and Communication Engineering, National Institute of Technology Puducherry, Karaikal, Puducherry-609609, India
<sup>c</sup> Department of Electronics and Communication Engineering, Manipal Institute of Technology Bengaluru, Manipal Academy of Higher Education, Manipal, Karnataka-576104, India
<sup>d</sup> Department of Computer Science and Engineering, Manipal Institute of Technology Bengaluru, Manipal Academy of Higher Education, Manipal, Karnataka-576104, India
<sup>e</sup> Manipal Institute of Technology, Manipal Academy of Higher Education (MAHE), Manipal, Udupi, Karnataka-576104, India

<sup>∗</sup> Corresponding author.

---

## Overview

Dermatological macro-photographs captured with ordinary digital cameras frequently suffer from uneven background illumination, which degrades lesion segmentation and therefore the diagnostic decisions of automated systems. Illumination correction is applied as a preprocessing step, but the existing no-reference image quality assessment (NR-IQA) metrics for this task — lightness-order error (LOE) and the average gradient of the illumination component (AGIC) — measure naturalness and residual illumination inhomogeneity, neither of which is guaranteed to track segmentation accuracy.

**GISR-Net** is a GoogLeNet-based image-to-scalar regressor trained with **task-based supervision**. Rather than subjective perceptual ratings, the training target is the mean Dice similarity index (DSI) achieved by three classical thresholding methods — Otsu, minimum error and Kapur maximum entropy — on the same image. The predicted score is termed the **GoogLeNet Illumination Correction Quality Index (GICQI)** and lies on the same [0, 1] scale as the Dice index.


## Key results

Evaluated under **source-image-disjoint** six-fold cross-validation on 480 held-out images from 30 photographs never seen during training:

| IQA metric | PLCC | SROCC | KROCC |
|---|---|---|---|
| LOE | 0.10 | 0.09 | 0.06 |
| AGIC | 0.35 | 0.08 | 0.05 |
| **GICQI (proposed)** | **0.79** | **0.72** | **0.53** |

GISR-Net attains MAE 0.114 ± 0.025 and RMSE 0.154 ± 0.035 with 5.60 M trainable parameters and an inference time of about 5 ms per image on a GPU. Ranking the 16 illumination-correction algorithms by mean predicted GICQI reproduces their ranking by mean true DSI with a Spearman correlation of 0.74, supporting the use of the metric to compare correction algorithms when expert masks are unavailable.

## Why the evaluation protocol matters

The 800 illumination-corrected images derive from only **50 original photographs** (16 correction algorithms applied to each). Any two images sharing a source photograph share the same lesion, skin, acquisition geometry and expert mask. Splitting such a corpus at the level of individual images lets a network recognise the photograph instead of judging the correction, which inflates measured performance.

Every partition here is therefore formed at the level of the **original photograph**:

- the 50 sources are shuffled with a fixed seed into 10 blocks of 5;
- fold *k* (k = 1…6) uses block *k* for testing and block *((k+6) mod 10)* for validation;
- each fold has **40 / 5 / 5 source photographs = 640 / 80 / 80 images**, an exact 80/10/10 split;
- the six test blocks are mutually disjoint, so 30 photographs (480 images) are tested, each exactly once;
- augmentation is applied to the training partition only.

`src/folds.py` asserts all of this before every training run — pairwise disjointness, completeness of the partition, exact image counts, and non-repetition of test blocks. `results/folds.json` records the exact photographs held out in each fold, so the split is independently reproducible.

## Installation

```bash
git clone https://github.com/VipinBioLab/GISR-Net.git
cd GISR-Net
pip install -r requirements.txt
```

Python 3.9–3.12. A CUDA GPU or Apple-Silicon MPS device is used automatically when available; the code also runs on CPU.

## Data

The dataset is not distributed with this repository. See [`data/README.md`](data/README.md) for the expected layout and file formats. Place `Data.csv` and the image folders under `data/`, or point the code elsewhere:

```bash
export GISRNET_ROOT=/path/to/your/data      # Linux / macOS
set GISRNET_ROOT=D:\path\to\your\data       # Windows
```

Baseline images come from the University of Waterloo skin cancer database; the external evaluation set comes from MED-NODE. Both must be obtained from their original providers under their own terms.

## Usage

### Score images with a trained model

```bash
python examples/predict.py --dir path/to/corrected_images \
    --checkpoint "results/models/gisrnet_tl_fold*.pt" --csv scores.csv
```

Averaging the six fold checkpoints is recommended; the script reports the standard deviation across folds alongside the mean score.

### Reproduce the full study

```bash
bash scripts/run_all.sh          # Linux / macOS
scripts\run_all.bat              # Windows (Command Prompt)
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1   # Windows (PowerShell)
```

This runs the fold construction and leakage check, builds the external test sets, trains GISR-Net and the four ablation configurations over six folds, evaluates generalisation, and regenerates every figure and table. It is **resumable**: folds already recorded in `results/results_<tag>.json` are skipped, so an interrupted run continues where it stopped.

Options: `EPOCHS=10` for a quick trial, `ACCUM=4` to fit a smaller GPU (the effective batch size stays 32).

A Colab notebook is provided in [`notebooks/`](notebooks/GISRNet_Colab.ipynb).

### Train a single configuration

```bash
python src/train.py --arch gisrnet --tag gisrnet_tl --auto-lr           # proposed model
python src/train.py --arch gisrnet --tag gisrnet_ri --auto-lr --no-pretrained
python src/train.py --arch gisrnet --tag gisrnet_noaug --auto-lr --no-augment
python src/train.py --arch gisrnet --tag gisrnet_mlp --auto-lr --head mlp
python src/train.py --arch gisrnet --tag gisrnet_sigmoid --auto-lr --head sigmoid
```

`--auto-lr` selects the initial learning rate with a short probe over a fixed grid, scored on the fold-1 validation partition only; the test partitions play no part in the choice. Everything else — optimiser, schedule shape, batch size, augmentation, folds and target — is held fixed, so the ablations isolate the component being varied.

> This repository contains the **proposed model only**. The comparison NR-IQA
> methods evaluated in the paper are the work of their respective authors; please
> obtain those implementations from the original sources.

## Repository layout

| Path | Contents |
|---|---|
| `src/config.py` | Paths, hyperparameters, correction-method names |
| `src/folds.py` | Source-disjoint fold construction and leakage assertions |
| `src/dataset.py` | Dataset, augmentation, on-disk image cache |
| `src/iqa_classical.py` | Otsu / minimum-error / Kapur thresholding, DSI, AGIC, LOE |
| `src/gisrnet.py` | GISR-Net model |
| `src/train.py` | Training driver |
| `src/build_external_sets.py` | Builds the external evaluation sets |
| `src/evaluate_external.py` | Generalisation evaluation |
| `src/make_figures.py`, `src/make_tables.py` | Figures and tables |
| `scripts/` | End-to-end runners for Linux/macOS and Windows |
| `notebooks/` | Colab notebook |
| `examples/predict.py` | Inference on new images |
| `results/` | Figures, tables and per-fold metrics from the reported run |

## Scope and limitations

- The target is defined by three **intensity-based** thresholding methods, so GICQI measures suitability for intensity-based lesion segmentation. It is not validated as a predictor of the accuracy of learned segmentation models.
- The model **regresses toward the mean**: the prediction-on-target slope is 0.53, and MAE is 0.076 on the lower-quality half of the test data against 0.152 on the higher-quality half. GISR-Net is better at flagging poor corrections than at grading good ones.
- **Per-image agreement is loose.** Bland–Altman limits span −0.34 to +0.26 Dice. The score is suited to ranking and screening, not to substituting for a measured Dice index on an individual image.
- **External generalisation is weak.** MAE roughly doubles on both external sets. On MED-NODE, where each image comes from a different photograph with a single correction applied, the variance is dominated by between-image differences rather than correction quality, and the classical metrics perform better there. GICQI is designed to rank alternative corrections *of a given photograph*.
- The development corpus rests on 50 photographs from a single database. Source-disjoint cross-validation removes leakage but cannot manufacture diversity the corpus does not contain.

## Citation

If you use this code or the GICQI metric, please cite:

```bibtex
@article{Venugopal_gisrnet,
  title   = {{GISR-Net}: An image-to-scalar regressor for quality assessment of
             illumination correction in dermatological macro-photographs},
  author  = {Venugopal, Vipin and Nath, Malaya Kumar and Joseph, Justin and
             Simi, V. R. and Anoop, B. N.},
  journal = {Elsevier (submitted)},
  year    = {2025},
  note    = {Manuscript submitted for publication}
}
```

> Update the `journal`, `volume`, `pages`, `year` and `doi` fields once the paper is accepted, and mirror the change in `CITATION.cff`.


## License

Code released under the MIT License — see [LICENSE](LICENSE). The datasets are subject to the licences of their original providers and are not covered by it.

## Acknowledgment

The authors thank Dr. M. Vipin Das, Department of Dermatology, Kerala Health Services, Trivandrum, Kerala, India, and Dr. Norton Stephen, Assistant Professor, Department of Pathology, All India Institute of Medical Sciences, Madurai, Tamil Nadu, India, for suggesting the need for an IQA metric to assess the quality of white balancing on dermatological macro-photographs, for assisting in rating the output quality of various white-balancing algorithms, and for ensuring the reliability of the gold-standard lesion segmentation.
