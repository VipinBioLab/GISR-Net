# Data

The datasets are **not distributed with this repository**. They must be obtained
from their original providers under their own terms:

| Source | Use | Where to obtain |
|---|---|---|
| University of Waterloo skin cancer database | 50 baseline photographs with uneven illumination, plus expert masks | <https://uwaterloo.ca/vision-image-processing-lab/research-demos/skin-cancer-detection> |
| MED-NODE | external cross-database evaluation set | <https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/> |

## Expected layout

Place the following under this `data/` folder, or set `GISRNET_ROOT` to the
folder that contains them:

```
data/
├── Data.csv                        development set index (see below)
├── HF_Test.csv                     external set A index
├── Enhanced_ECCA/                  16 folders of illumination-corrected images,
├── Enhanced_Fusion/                one per correction algorithm,
├── Enhanced_GCCA/                  50 images each  (50 x 16 = 800 images)
├── ...
├── Enhanced_Zhou/
├── GT/                             expert masks for the 50 baseline photographs
├── HF_Test/                        external set A images
├── Enhanced_MedNodeTest/           external set B, illumination-corrected
├── Simulated_MED_NODE_Test/        external set B, before correction (for LOE)
└── MED_NODE_Test_GT/               external set B expert masks
```

## `Data.csv`

One row per illumination-corrected image, 800 rows plus a header.

```csv
Name,DSI,AGIC,LOE
Enhanced_ECCA\01_Orig_ECCA.jpg,0.250232071,0.30310256,498.8461538
Enhanced_ECCA\02_Orig_ECCA.jpg,0.066859809,0.345167397,39.21575985
```

| Column | Meaning |
|---|---|
| `Name` | Path relative to the data root. Backslashes and forward slashes are both accepted. |
| `DSI` | Training target: the mean Dice index of Otsu, minimum error and Kapur segmentation against the expert mask. |
| `AGIC` | Average gradient of the illumination component, for comparison. |
| `LOE` | Lightness-order error, for comparison. |

**The file-name convention carries the split.** The source photograph is taken to
be the leading token of the file name before the first underscore — `01` in
`01_Orig_ECCA.jpg`. All 16 corrected versions of a photograph therefore share a
source identifier, and `src/folds.py` uses it to keep them together in one
partition. If your file names follow a different convention, adjust
`source_id_of()` in `src/folds.py` accordingly, or the leakage guarantee will
not hold.

## `HF_Test.csv`

```csv
Images,Original_DSI
HF_Test\001_HF.jpg,0.2
HF_Test\002_HF.jpg,0.42
```

## External set B (MED-NODE)

Targets for this set are **computed by the code**, not supplied. Running

```bash
python src/build_external_sets.py
```

applies Otsu, minimum error and Kapur segmentation to each image in
`Enhanced_MedNodeTest/`, compares the results with the masks in
`MED_NODE_Test_GT/`, and writes `results/external_mednode_test.csv` containing
the mean DSI target together with AGIC and LOE. Files are matched by the leading
numeric token, so `443_OrigMEdNodeTest_HF.jpg` pairs with `443_GT.png` and with
`443_OrigMEdNodeTest.jpg` in `Simulated_MED_NODE_Test/`.

## Regenerating the DSI targets

`src/iqa_classical.py` contains standalone implementations of Otsu, minimum
error (Kittler–Illingworth) and Kapur maximum entropy thresholding, the Dice
index, AGIC and LOE. Use `mean_dsi(image_path, gt_path)` to recompute a target
for any image–mask pair. The lesion is taken to be the sub-threshold class,
since lesions appear darker than surrounding skin on macro-photographs; this
polarity is fixed a priori and is never selected using the ground truth.
