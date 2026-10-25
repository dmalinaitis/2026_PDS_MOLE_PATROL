"""
Shortcut removal: hair and pen marks are detected and painted over (OpenCV Telea inpainting)
before the colour features are extracted, so they cannot act as shortcuts for the model.

Five annotators rated the images in data/annotations_combined.csv for hair (0-3) and pen
marks (0/1). A detector is only skipped when the annotators agree it is not needed, so
images without a trusted annotation always go through both detectors.
"""
import cv2
import numpy as np
import pandas as pd

from utils import ANNOTATIONS_CSV_PATH, normalize_img_id

HAIR_COLUMNS = [f"hair_{i}" for i in range(1, 6)]
PEN_COLUMNS = [f"pen_{i}" for i in range(1, 6)]
MIN_VOTES = 4  # an average rating is only trusted if at least 4 of the 5 ratings are valid


def valid_ratings(ratings, valid_values):
    """Ratings as numbers, with anything outside valid_values set to NaN."""
    ratings = ratings.apply(pd.to_numeric, errors="coerce")
    return ratings.where(ratings.isin(valid_values))


def load_shortcut_annotations(path=ANNOTATIONS_CSV_PATH):
    """Average hair score (0-2, a rating of 3 counts as 2) and pen score (0-1) per image.

    Indexed by img_id. A score is NaN when fewer than MIN_VOTES of its ratings are valid.
    """
    annotations = pd.read_csv(path)
    hair = valid_ratings(annotations[HAIR_COLUMNS], {0, 1, 2, 3}).replace({3: 2})
    pen = valid_ratings(annotations[PEN_COLUMNS], {0, 1})
    scores = pd.DataFrame({
        "img_id": annotations["img_id"].apply(normalize_img_id),
        "hair_score": hair.mean(axis=1).where(hair.notna().sum(axis=1) >= MIN_VOTES),
        "pen_score": pen.mean(axis=1).where(pen.notna().sum(axis=1) >= MIN_VOTES),
    })
    scores = scores[scores["img_id"] != "nan"]  # rows without an image id
    return scores.drop_duplicates(subset="img_id", keep="first").set_index("img_id")


def get_scores(img_id, annotations):
    """(hair_score, pen_score) of one image; NaN when it has no trusted annotation."""
    if img_id in annotations.index:
        return annotations.at[img_id, "hair_score"], annotations.at[img_id, "pen_score"]
    return np.nan, np.nan


def hair_level(hair_score):
    """'auto' (no trusted annotation), 'skip' (score <= 0.5), 'moderate' (<= 1.25) or 'heavy'."""
    if pd.isna(hair_score):
        return "auto"
    if hair_score <= 0.5:
        return "skip"
    if hair_score <= 1.25:
        return "moderate"
    return "heavy"


def detect_hair_mask(image_bgr):
    """Mask of thin dark structures (hairs), found with a black-hat filter."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # smooth first so skin texture is not detected
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)  # dark details thinner than the kernel
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # remove small noisy dots
    return cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)  # grow slightly to cover the hairs


def detect_pen_mask(image_bgr):
    """Mask of blue/purple pen marks, found by HSV colour thresholding."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([95, 25, 20], dtype=np.uint8), np.array([175, 255, 230], dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # close gaps in the strokes
    return cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)


def apply_shortcuts(image_rgb, img_id, annotations):
    """Remove hair and pen marks from an RGB image.

    The hair detector runs unless the trusted hair score is 0.5 or less ('skip'), and the pen
    detector runs unless the trusted pen score is below 0.5. Everything either detector finds
    is inpainted. Returns (cleaned_image, hair_mask, pen_mask).
    """
    hair_score, pen_score = get_scores(img_id, annotations)
    image_bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb[:, :, :3]), cv2.COLOR_RGB2BGR)
    no_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    hair_mask = detect_hair_mask(image_bgr) if hair_level(hair_score) != "skip" else no_mask
    pen_mask = detect_pen_mask(image_bgr) if pd.isna(pen_score) or pen_score >= 0.5 else no_mask
    cleaned = cv2.inpaint(image_bgr, cv2.bitwise_or(hair_mask, pen_mask), 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB), hair_mask, pen_mask
