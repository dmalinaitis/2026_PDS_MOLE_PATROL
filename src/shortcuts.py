import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from utils import ANNOTATIONS_CSV_PATH


HAIR_COLUMNS = [f"hair_{idx}" for idx in range(1, 6)]
PEN_COLUMNS = [f"pen_{idx}" for idx in range(1, 6)]
MIN_ANNOTATION_VOTES = 4


@dataclass(frozen=True)
class ShortcutDecision:
    img_id: str
    source: str
    hair_score: Optional[float]
    hair_level: str
    hair_votes: int
    hair_source: str
    pen_score: Optional[float]
    pen_present: Optional[bool]
    pen_votes: int
    pen_source: str


@dataclass(frozen=True)
class ShortcutResult:
    img_id: str
    cleaned_image: np.ndarray
    hair_mask: np.ndarray
    pen_mask: np.ndarray
    shortcut_mask: np.ndarray
    decision: ShortcutDecision
    hair_applied: bool
    pen_applied: bool
    mask_coverage: float
    rejected_auto_mask: bool


def normalize_img_id(img_id):
    """Return an image id without a directory or file extension."""
    return os.path.splitext(os.path.basename(str(img_id)))[0]


def _coerce_ratings(frame, columns, valid_values):
    ratings = frame[columns].apply(pd.to_numeric, errors="coerce")
    return ratings.where(ratings.isin(valid_values))


def _hair_level(score):
    if pd.isna(score):
        return "auto"
    if score <= 0.5:
        return "skip"
    if score <= 1.25:
        return "moderate"
    return "heavy"


def load_shortcut_annotations(
    annotations_path=ANNOTATIONS_CSV_PATH,
    min_votes=MIN_ANNOTATION_VOTES,
):
    """Load human shortcut annotations and convert them into brushing gates."""
    annotations = pd.read_csv(annotations_path)
    required_columns = {"img_id", *HAIR_COLUMNS, *PEN_COLUMNS}
    missing_columns = required_columns - set(annotations.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"annotations file is missing required columns: {missing}")

    cleaned = pd.DataFrame()
    cleaned["img_id"] = annotations["img_id"].apply(normalize_img_id)

    hair = _coerce_ratings(annotations, HAIR_COLUMNS, {0, 1, 2, 3}).replace({3: 2})
    pen = _coerce_ratings(annotations, PEN_COLUMNS, {0, 1})

    cleaned["hair_votes"] = hair.notna().sum(axis=1).astype(int)
    cleaned["hair_score"] = hair.mean(axis=1)
    cleaned.loc[cleaned["hair_votes"] < min_votes, "hair_score"] = pd.NA
    cleaned["hair_level"] = cleaned["hair_score"].apply(_hair_level)
    cleaned["hair_source"] = np.where(
        cleaned["hair_votes"] >= min_votes,
        "annotation",
        "auto",
    )

    cleaned["pen_votes"] = pen.notna().sum(axis=1).astype(int)
    cleaned["pen_score"] = pen.mean(axis=1)
    cleaned.loc[cleaned["pen_votes"] < min_votes, "pen_score"] = pd.NA
    cleaned["pen_present"] = (cleaned["pen_score"] >= 0.5).astype(object)
    cleaned.loc[cleaned["pen_score"].isna(), "pen_present"] = None
    cleaned["pen_source"] = np.where(
        cleaned["pen_votes"] >= min_votes,
        "annotation",
        "auto",
    )

    cleaned = cleaned[cleaned["img_id"].notna() & (cleaned["img_id"] != "nan")]
    cleaned = cleaned.drop_duplicates(subset="img_id", keep="first")
    return cleaned.set_index("img_id", drop=False)


def get_shortcut_decision(img_id, annotations=None):
    """Return annotation-driven or auto gates for one image."""
    normalized_img_id = normalize_img_id(img_id)
    if annotations is None:
        annotations = load_shortcut_annotations()

    if normalized_img_id not in annotations.index:
        return ShortcutDecision(
            img_id=normalized_img_id,
            source="auto",
            hair_score=None,
            hair_level="auto",
            hair_votes=0,
            hair_source="auto",
            pen_score=None,
            pen_present=None,
            pen_votes=0,
            pen_source="auto",
        )

    row = annotations.loc[normalized_img_id]
    hair_source = str(row["hair_source"])
    pen_source = str(row["pen_source"])
    if hair_source == "annotation" and pen_source == "annotation":
        source = "annotation"
    elif hair_source == "auto" and pen_source == "auto":
        source = "auto"
    else:
        source = "mixed"

    hair_score = row["hair_score"]
    pen_score = row["pen_score"]
    pen_present = row["pen_present"]

    return ShortcutDecision(
        img_id=normalized_img_id,
        source=source,
        hair_score=None if pd.isna(hair_score) else float(hair_score),
        hair_level=str(row["hair_level"]),
        hair_votes=int(row["hair_votes"]),
        hair_source=hair_source,
        pen_score=None if pd.isna(pen_score) else float(pen_score),
        pen_present=None if pd.isna(pen_present) else bool(pen_present),
        pen_votes=int(row["pen_votes"]),
        pen_source=pen_source,
    )


def detect_pen_mask(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_pen = np.array([95, 25, 20], dtype=np.uint8)
    upper_pen = np.array([175, 255, 230], dtype=np.uint8)

    pen_mask = cv2.inRange(hsv, lower_pen, upper_pen)

    pen_mask = cv2.morphologyEx(
        pen_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
    )

    pen_mask = cv2.dilate(
        pen_mask,
        np.ones((5, 5), np.uint8),
        iterations=2,
    )

    return pen_mask


def detect_hair_mask(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Smooth a bit first so skin texture is less detected
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect dark thin structures
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    blackhat = cv2.morphologyEx(gray_blur, cv2.MORPH_BLACKHAT, blackhat_kernel)

    # Higher threshold = less white noise
    _, hair_mask = cv2.threshold(
        blackhat,
        10,
        255,
        cv2.THRESH_BINARY,
    )

    # Remove small noisy dots
    hair_mask = cv2.morphologyEx(
        hair_mask,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
    )

    # Small dilation so detected hairs are covered
    hair_mask = cv2.dilate(
        hair_mask,
        np.ones((2, 2), np.uint8),
        iterations=1,
    )

    return hair_mask


def build_shortcut_masks(image_bgr, decision):
    empty_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    if decision.hair_source == "auto" or decision.hair_level in {"moderate", "heavy"}:
        hair_mask = detect_hair_mask(image_bgr)
    else:
        hair_mask = empty_mask.copy()

    if decision.pen_source == "auto" or decision.pen_present is True:
        pen_mask = detect_pen_mask(image_bgr)
    else:
        pen_mask = empty_mask.copy()

    shortcut_mask = cv2.bitwise_or(hair_mask, pen_mask)
    return hair_mask, pen_mask, shortcut_mask


def apply_shortcuts(image_rgb, img_id=None, annotations=None):
    """Apply the original shortcut notebook logic to an RGB image."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        raise ValueError("image_rgb must be an RGB image with shape HxWx3")

    decision = get_shortcut_decision(img_id or "", annotations=annotations)
    image_rgb = np.ascontiguousarray(image_rgb[:, :, :3])
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    hair_mask, pen_mask, shortcut_mask = build_shortcut_masks(image_bgr, decision)

    cleaned_bgr = cv2.inpaint(
        image_bgr,
        shortcut_mask,
        3,
        cv2.INPAINT_TELEA,
    )
    cleaned_image = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
    mask_coverage = float(np.count_nonzero(shortcut_mask) / shortcut_mask.size)

    return ShortcutResult(
        img_id=decision.img_id,
        cleaned_image=cleaned_image,
        hair_mask=hair_mask,
        pen_mask=pen_mask,
        shortcut_mask=shortcut_mask,
        decision=decision,
        hair_applied=bool(np.any(hair_mask)),
        pen_applied=bool(np.any(pen_mask)),
        mask_coverage=mask_coverage,
        rejected_auto_mask=False,
    )
