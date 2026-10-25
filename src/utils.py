"""Paths and feature column names shared by the extraction code and the figures notebook."""
import os

# Every path is built from the project root (the folder above src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMGS_DIR = os.path.join(DATA_DIR, "imgs")
MASK_DIR = os.path.join(DATA_DIR, "masks")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
METADATA_CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")
ANNOTATIONS_CSV_PATH = os.path.join(DATA_DIR, "annotations_combined.csv")
BASELINE_FEATURES_CSV_PATH = os.path.join(DATA_DIR, "baseline_features.csv")
EXTENDED_FEATURES_CSV_PATH = os.path.join(DATA_DIR, "extended_features.csv")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# The six statistics computed for every colour channel (see feature_C.py)
STATISTICS = ("mean", "std", "skew", "5p", "50p", "95p")
HSV_COLUMNS = [f"{channel}_{stat}" for channel in ("hue", "saturation", "value") for stat in STATISTICS]
RGB_GREY_COLUMNS = [f"{channel}_{stat}" for channel in ("red", "green", "blue", "grey") for stat in STATISTICS]

# Column order of the two feature tables. The order matters: the models use every column
# from asymmetry_score to border_score as input (see src/model.py).
BASELINE_FEATURE_COLUMNS = ["patient_id", "diagnostic", "img_id", "asymmetry_score", "border_score", *HSV_COLUMNS]
EXTENDED_FEATURE_COLUMNS = [
    "patient_id", "diagnostic", "img_id", "asymmetry_score", *HSV_COLUMNS, *RGB_GREY_COLUMNS, "border_score",
]


def normalize_img_id(img_id):
    """Image id without folder or file extension, e.g. 'imgs/PAT_8_15_820.png' -> 'PAT_8_15_820'."""
    return os.path.splitext(os.path.basename(str(img_id)))[0]
