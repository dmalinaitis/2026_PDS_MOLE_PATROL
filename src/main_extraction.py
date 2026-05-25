import os

import cv2
import pandas as pd

from feature_A import get_asymmetry
from feature_B import get_compactness
from feature_C import get_color_data
from utils import (
    BASELINE_FEATURE_COLUMNS,
    BASELINE_FEATURES_CSV_PATH,
    EXTENDED_FEATURES_CSV_PATH,
    IMAGE_EXTENSIONS,
    IMGS_DIR,
    MASK_DIR,
    MASK_EXTENSIONS,
    METADATA_CSV_PATH,
    TESTING_BASELINE_FEATURES_CSV_PATH,
    TESTING_BASELINE_SUBSET,
    TESTING_EXTENDED_FEATURES_CSV_PATH,
)


def normalize_img_id(img_id):
    """Return the image id without file extension."""
    return os.path.splitext(os.path.basename(str(img_id)))[0]


def find_file(directory, stem, suffix="", extensions=IMAGE_EXTENSIONS):
    for extension in extensions:
        path = os.path.join(directory, f"{stem}{suffix}{extension}")
        if os.path.exists(path):
            return path
    return None


def has_image_and_mask(img_id):
    image_path = find_file(IMGS_DIR, img_id, extensions=IMAGE_EXTENSIONS)
    mask_path = find_file(MASK_DIR, img_id, suffix="_mask", extensions=MASK_EXTENSIONS)
    return image_path is not None and mask_path is not None


def has_nonempty_mask(img_id):
    mask_path = find_file(MASK_DIR, img_id, suffix="_mask", extensions=MASK_EXTENSIONS)
    if mask_path is None:
        return False

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask is not None and (mask > 0).any()


def has_usable_image_and_mask(img_id):
    image_path = find_file(IMGS_DIR, img_id, extensions=IMAGE_EXTENSIONS)
    mask_path = find_file(MASK_DIR, img_id, suffix="_mask", extensions=MASK_EXTENSIONS)
    if image_path is None or mask_path is None:
        return False

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        return False
    if not (mask > 0).any():
        return False

    return image.shape[:2] == mask.shape


def load_testing_baseline_subset():
    subset = pd.DataFrame(TESTING_BASELINE_SUBSET)
    subset["img_id"] = subset["img_id"].apply(normalize_img_id)
    return subset[["patient_id", "diagnostic", "img_id"]]


def load_baseline_dataset_rows():
    metadata = pd.read_csv(METADATA_CSV_PATH)
    required_columns = {"patient_id", "diagnostic", "img_id"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"metadata.csv is missing required columns: {missing}")

    rows = metadata[["patient_id", "diagnostic", "img_id"]].copy()
    rows["img_id"] = rows["img_id"].apply(normalize_img_id)
    rows = rows[rows["img_id"].apply(has_usable_image_and_mask)]
    return rows.sort_values("img_id").reset_index(drop=True)


def load_image_and_mask(img_id):
    image_path = find_file(IMGS_DIR, img_id, extensions=IMAGE_EXTENSIONS)
    mask_path = find_file(MASK_DIR, img_id, suffix="_mask", extensions=MASK_EXTENSIONS)

    if image_path is None:
        raise FileNotFoundError(f"No image found for img_id: {img_id}")
    if mask_path is None:
        raise FileNotFoundError(f"No mask found for img_id: {img_id}")

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image_bgr is None:
        raise ValueError(f"Unable to read image file: {image_path}")
    if mask is None:
        raise ValueError(f"Unable to read mask file: {mask_path}")
    if not (mask > 0).any():
        raise ValueError(f"Mask contains no lesion pixels for img_id: {img_id}")
    if image_bgr.shape[:2] != mask.shape:
        raise ValueError(
            f"Image and mask shapes do not match for {img_id}: "
            f"image={image_bgr.shape[:2]}, mask={mask.shape}"
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, mask


def extract_features_for_row(row):
    img_id = normalize_img_id(row["img_id"])
    image, mask = load_image_and_mask(img_id)

    features = {
        "patient_id": row["patient_id"],
        "diagnostic": row["diagnostic"],
        "img_id": img_id,
        "asymmetry_score": get_asymmetry(mask),
        "border_score": get_compactness(mask),
    }
    features.update(get_color_data(image, mask))
    return features


def write_feature_batch(batch, output_path, include_header):
    features = pd.DataFrame(batch).reindex(columns=BASELINE_FEATURE_COLUMNS)
    write_mode = "w" if include_header else "a"
    features.to_csv(output_path, mode=write_mode, header=include_header, index=False)


def extract_features(rows, output_path, batch_size=50):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    results = []
    batch = []
    total_rows = len(rows)
    wrote_header = False

    for processed_rows, (_, row) in enumerate(rows.iterrows(), start=1):
        result = extract_features_for_row(row)
        results.append(result)
        batch.append(result)

        if len(batch) >= batch_size:
            write_feature_batch(batch, output_path, include_header=not wrote_header)
            wrote_header = True
            batch = []
            print(f"Processed {processed_rows}/{total_rows} rows", flush=True)

    if batch:
        write_feature_batch(batch, output_path, include_header=not wrote_header)

    features = pd.DataFrame(results).reindex(columns=BASELINE_FEATURE_COLUMNS)
    print(f"Saved {len(features)} feature rows to {output_path}")
    return features


def extract_testing_baseline_features(output_path=TESTING_BASELINE_FEATURES_CSV_PATH):
    rows = load_testing_baseline_subset()
    return extract_features(rows, output_path)


def extract_baseline_features(output_path=BASELINE_FEATURES_CSV_PATH):
    rows = load_baseline_dataset_rows()
    return extract_features(rows, output_path)


#TODO for the future: once the shortcut/brushing script creates cleaned images,
# load those cleaned images here and write the same feature columns to
# testing_extended_features.csv.
def extract_testing_extended_features(output_path=TESTING_EXTENDED_FEATURES_CSV_PATH):
    raise NotImplementedError("TODO: implement after the shortcut/brushing script exists.")


#TODO for the future: once the shortcut/brushing script creates cleaned images,
# extract all valid cleaned-image/mask pairs and write extended_features.csv.
def extract_extended_features(output_path=EXTENDED_FEATURES_CSV_PATH):
    raise NotImplementedError("TODO: implement after the shortcut/brushing script exists.")


def extract_testing_features():
    """Backward-compatible alias for the testing baseline extraction."""
    return extract_testing_baseline_features()


def extract_all_features():
    """Backward-compatible alias for full baseline extraction."""
    return extract_baseline_features()


if __name__ == "__main__":
    extract_baseline_features()
