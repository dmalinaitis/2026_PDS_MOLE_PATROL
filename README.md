# The Mole Patrol - skin cancer classification

Final project for *Projects in Data Science* (2nd semester, 2026) at the IT University of Copenhagen,
by Bartosz Kaluski, David Malinaitis, David Moreno Ollero, Jamie Underwood and Rui Zhu.
The full write-up is in [`results/reports/PDS_2026_MOLE_PATROL_REPORT.pdf`](results/reports/PDS_2026_MOLE_PATROL_REPORT.pdf).

The project classifies skin lesion images from the [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)
dataset as **cancer** (BCC, SCC, MEL) or **not cancer** (ACK, NEV, SEK) using interpretable ABC features:

- **A**symmetry: how much the lesion mask differs from its mirror image (`src/feature_A.py`)
- **B**order: compactness of the lesion mask (`src/feature_B.py`)
- **C**olour: statistics of the lesion pixels in HSV, RGB and greyscale (`src/feature_C.py`)

## Results

Accuracy and AUC on the 401 held-out validation images. Every model is a majority vote of 5 Random
Forests (200 trees, max depth 4), one per cross-validation fold. All models use asymmetry and border;
they differ in their colour features. Every model except `base` uses the extended features, where hair
and pen marks were removed from the images before the colour features were extracted.

| Model      | Colour features       | Accuracy |    AUC |
|------------|-----------------------|---------:|-------:|
| `base`     | none (see notes)      |   0.5661 | 0.5656 |
| `extended` | HSV                   |   0.7257 | 0.7250 |
| `grey`     | greyscale             |   0.6234 | 0.6229 |
| `rgb`      | RGB                   |   0.6683 | 0.6686 |
| `rgb_grey` | RGB + greyscale       |   0.6584 | 0.6586 |
| `hsv_grey` | HSV + greyscale       |   **0.7382** | **0.7376** |
| `rgb_hsv`  | RGB + HSV             |   0.7232 | 0.7226 |
| `all`      | HSV + RGB + greyscale |   0.7207 | 0.7203 |

## Pipeline

1. **Feature extraction** (`src/main_extraction.py`) builds `data/baseline_features.csv` from the original
   images and `data/extended_features.csv` from images cleaned by `src/shortcuts.py` (hair and pen marks
   detected and inpainted), adding RGB and greyscale statistics. 203 of the 2,298 images have no usable
   mask, which leaves 2,095.
2. **Split** (`src/model.py`): the 222 patients listed in `data/Validation.csv` (440 images, about 20%)
   are held out. The rest is split into 5 folds with `StratifiedGroupKFold`: grouped by patient, so no
   patient is in both the training and test data, and stratified by diagnosis.
3. **Training**: one Random Forest per fold. The 5 forests of each model are saved together in `results/models/`.
4. **Validation**: the 5 forests vote on each validation image (cancer if at least 3 agree). The
   predictions are saved in `results/predictions/`.

## How to run

Tested with Python 3.14. Run everything from the project root.

```bash
pip install -r requirements.txt

python main.py                 # evaluate the saved models (prints the table above)
python main.py --train         # retrain all models from scratch, overwriting results/models/
python src/main_extraction.py  # rebuild the two feature tables (needs the images, see below)
```

The report figures can be recreated with `src/plots_visualised.ipynb`.

The PAD-UFES-20 images and masks are not part of the repository. To rebuild the feature tables, put the
images in `data/imgs/` (`PAT_<...>.png`) and the masks in `data/masks/` (`PAT_<...>_mask.png`).

**Retraining needs scikit-learn 1.7.2** (pinned in `requirements.txt`). Newer versions assign patients to
the folds differently, so retrained models would no longer match the saved models or the scores above.

## Repository layout

```
├── main.py                       trains or loads the 8 models, evaluates them, saves predictions
├── requirements.txt
├── data/
│   ├── metadata.csv              PAD-UFES-20 metadata (2,298 images, 1,373 patients)
│   ├── Validation.csv            images reserved for validation
│   ├── annotations_combined.csv  hair and pen-mark ratings by 5 annotators
│   ├── baseline_features.csv     features of the original images
│   ├── extended_features.csv     features after hair/pen removal, incl. RGB and greyscale
│   ├── imgs/, masks/             PAD-UFES-20 images and lesion masks (not committed)
├── src/
│   ├── feature_A.py              asymmetry
│   ├── feature_B.py              border compactness
│   ├── feature_C.py              colour statistics
│   ├── shortcuts.py              hair and pen-mark removal
│   ├── main_extraction.py        builds the feature tables
│   ├── model.py                  labels, split, training and evaluation
│   ├── utils.py                  shared paths and column names
│   └── plots_visualised.ipynb    report figures
└── results/
    ├── figures/                  figures used in the report
    ├── models/                   saved models (5 Random Forests each)
    ├── predictions/              validation predictions of every model
    └── reports/                  final report
```

## Notes

- **The `base` model uses no colour features.** `baseline_features.csv` contains HSV statistics, but the
  models use the columns from `asymmetry_score` to `border_score`, and in the baseline table those two
  columns are next to each other. This is left as it is so the results match the report.
- The AUC is computed from the hard 0/1 votes, not from probabilities.
- `Validation.csv` lists 440 images, but only the 401 with a usable mask have features and predictions.
