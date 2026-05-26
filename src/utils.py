import os

# Get the directory where utils.py is located (the 'src' folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to get the project root directory
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Define paths relative to the project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MASK_DIR = os.path.join(PROJECT_ROOT, "data", "masks")
IMGS_DIR = os.path.join(PROJECT_ROOT, "data", "imgs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METADATA_CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")
ANNOTATIONS_CSV_PATH = os.path.join(DATA_DIR, "annotations_combined.csv")
TESTING_BASELINE_FEATURES_CSV_PATH = os.path.join(
    DATA_DIR, "testing_baseline_features.csv"
)
BASELINE_FEATURES_CSV_PATH = os.path.join(DATA_DIR, "baseline_features.csv")
TESTING_EXTENDED_FEATURES_CSV_PATH = os.path.join(
    DATA_DIR, "testing_extended_features.csv"
)
EXTENDED_FEATURES_CSV_PATH = os.path.join(DATA_DIR, "extended_features.csv")

# Backwards-compatible names used by older extraction code/notebooks.
TESTING_FEATURES_CSV_PATH = TESTING_BASELINE_FEATURES_CSV_PATH
FEATURES_CSV_PATH = BASELINE_FEATURES_CSV_PATH
OUTPUT_CSV_PATH = TESTING_BASELINE_FEATURES_CSV_PATH

TEST_SUBSET_SIZE = 30
TEST_SUBSET_SEED = 11037

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg")

COLOR_FEATURE_COLUMNS = [
    "hue_mean",
    "hue_std",
    "hue_skew",
    "hue_5p",
    "hue_50p",
    "hue_95p",
    "saturation_mean",
    "saturation_std",
    "saturation_skew",
    "saturation_5p",
    "saturation_50p",
    "saturation_95p",
    "value_mean",
    "value_std",
    "value_skew",
    "value_5p",
    "value_50p",
    "value_95p",
]

BASELINE_FEATURE_COLUMNS = [
    "patient_id",
    "diagnostic",
    "img_id",
    "asymmetry_score",
    "border_score",
    *COLOR_FEATURE_COLUMNS,
]

TESTING_FEATURE_COLUMNS = BASELINE_FEATURE_COLUMNS

TESTING_BASELINE_SUBSET = [
    {"patient_id": "PAT_1186", "diagnostic": "NEV", "img_id": "PAT_1186_680_578"},
    {"patient_id": "PAT_1259", "diagnostic": "ACK", "img_id": "PAT_1259_892_793"},
    {"patient_id": "PAT_1631", "diagnostic": "ACK", "img_id": "PAT_1631_2837_667"},
    {"patient_id": "PAT_165", "diagnostic": "BCC", "img_id": "PAT_165_256_460"},
    {"patient_id": "PAT_1689", "diagnostic": "ACK", "img_id": "PAT_1689_3075_38"},
    {"patient_id": "PAT_1712", "diagnostic": "ACK", "img_id": "PAT_1712_3173_684"},
    {"patient_id": "PAT_1801", "diagnostic": "NEV", "img_id": "PAT_1801_3459_857"},
    {"patient_id": "PAT_1850", "diagnostic": "ACK", "img_id": "PAT_1850_3631_519"},
    {"patient_id": "PAT_187", "diagnostic": "BCC", "img_id": "PAT_187_287_482"},
    {"patient_id": "PAT_257", "diagnostic": "BCC", "img_id": "PAT_257_396_494"},
    {"patient_id": "PAT_290", "diagnostic": "BCC", "img_id": "PAT_290_445_686"},
    {"patient_id": "PAT_304", "diagnostic": "ACK", "img_id": "PAT_304_4186_458"},
    {"patient_id": "PAT_422", "diagnostic": "BCC", "img_id": "PAT_422_837_683"},
    {"patient_id": "PAT_473", "diagnostic": "BCC", "img_id": "PAT_473_911_610"},
    {"patient_id": "PAT_503", "diagnostic": "BCC", "img_id": "PAT_503_950_552"},
    {"patient_id": "PAT_625", "diagnostic": "SEK", "img_id": "PAT_625_1184_994"},
    {"patient_id": "PAT_639", "diagnostic": "ACK", "img_id": "PAT_639_3674_452"},
    {"patient_id": "PAT_667", "diagnostic": "BCC", "img_id": "PAT_667_1264_8"},
    {"patient_id": "PAT_721", "diagnostic": "BCC", "img_id": "PAT_721_1365_819"},
    {"patient_id": "PAT_727", "diagnostic": "BCC", "img_id": "PAT_727_1373_776"},
    {"patient_id": "PAT_741", "diagnostic": "BCC", "img_id": "PAT_741_1400_780"},
    {"patient_id": "PAT_771", "diagnostic": "BCC", "img_id": "PAT_771_1490_334"},
    {"patient_id": "PAT_819", "diagnostic": "SEK", "img_id": "PAT_819_1541_727"},
    {"patient_id": "PAT_834", "diagnostic": "SCC", "img_id": "PAT_834_1574_276"},
    {"patient_id": "PAT_850", "diagnostic": "BCC", "img_id": "PAT_850_1611_332"},
    {"patient_id": "PAT_856", "diagnostic": "BCC", "img_id": "PAT_856_1626_523"},
    {"patient_id": "PAT_904", "diagnostic": "BCC", "img_id": "PAT_904_1736_851"},
    {"patient_id": "PAT_937", "diagnostic": "ACK", "img_id": "PAT_937_1768_126"},
    {"patient_id": "PAT_93", "diagnostic": "SEK", "img_id": "PAT_93_361_231"},
    {"patient_id": "PAT_986", "diagnostic": "BCC", "img_id": "PAT_986_1855_702"},
]

TEST_SUBSET = [row["img_id"] for row in TESTING_BASELINE_SUBSET]
