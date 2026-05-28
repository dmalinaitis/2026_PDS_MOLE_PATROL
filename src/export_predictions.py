import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.cancer
import src.drop


DATA_DIR = Path("data")
MODELS_DIR = Path("results") / "models"
PREDICTIONS_DIR = Path("results") / "predictions"

MODEL_CONFIGS = [
    ("base", "basemodel.pkl", "predictions_base.csv"),
    ("extended", "extendedmodel.pkl", "predictions_extended.csv"),
    ("grey", "greymodel.pkl", "predictions_grey.csv"),
    ("rgb", "rgbmodel.pkl", "predictions_rgb.csv"),
    ("rgb_grey", "rgbgreymodel.pkl", "predictions_rgb_grey.csv"),
    ("hsv_grey", "hsvgreymodel.pkl", "predictions_hsv_grey.csv"),
    ("rgb_hsv", "rgbhsvmodel.pkl", "predictions_rgb_hsv.csv"),
    ("all", "allcolormodel.pkl", "predictions_all.csv"),
]


def load_model_array(model_filename):
    with open(MODELS_DIR / model_filename, "rb") as model_file:
        return pickle.load(model_file)


def prepare_features(features):
    prepared = features.copy()
    prepared["diagnostic"] = prepared["diagnostic"].apply(src.cancer.cancer).astype(int)
    return prepared


def build_feature_variants(extended_features):
    rgb_grey_features = src.drop.hsv(extended_features)
    grey_features = src.drop.rgb(rgb_grey_features)
    rgb_hsv_features = src.drop.grey(extended_features)
    hsv_features = src.drop.rgb(rgb_hsv_features)
    rgb_features = src.drop.grey(rgb_grey_features)
    hsv_grey_features = src.drop.rgb(extended_features)

    return {
        "extended": hsv_features,
        "grey": grey_features,
        "rgb": rgb_features,
        "rgb_grey": rgb_grey_features,
        "hsv_grey": hsv_grey_features,
        "rgb_hsv": rgb_hsv_features,
        "all": extended_features,
    }


def get_prediction_features(validation_features, model_array):
    feature_names = getattr(model_array[0], "feature_names_in_", None)
    if feature_names is not None:
        return validation_features.loc[:, list(feature_names)]

    return validation_features.loc[:, "asymmetry_score":"border_score"]


def majority_vote_predictions(model_array, prediction_features):
    model_predictions = []
    for model in model_array:
        predictions = model.predict(prediction_features)
        model_predictions.append([int(prediction) for prediction in predictions])

    majority_predictions = []
    for row_predictions in zip(*model_predictions):
        majority_predictions.append(1 if sum(row_predictions) >= 3 else 0)

    return majority_predictions


def export_model_predictions(model_name, model_array, features, validation_patient_ids, output_filename):
    validation_features = features[features["patient_id"].isin(validation_patient_ids)]
    validation_features = validation_features.reset_index(drop=True)
    prediction_features = get_prediction_features(validation_features, model_array)
    predictions = majority_vote_predictions(model_array, prediction_features)

    results = pd.DataFrame(
        {
            "img_id": validation_features["img_id"],
            "patient_id": validation_features["patient_id"],
            "true_label": validation_features["diagnostic"].astype(int),
            "predicted_label": predictions,
        }
    )

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(PREDICTIONS_DIR / output_filename, index=False)

    accuracy = (results["true_label"] == results["predicted_label"]).mean()
    print(f"{model_name}: saved {len(results)} rows to {PREDICTIONS_DIR / output_filename}")
    print(f"{model_name}: accuracy = {accuracy:.4f}")


def main():
    baseline_features = prepare_features(pd.read_csv(DATA_DIR / "baseline_features.csv"))
    extended_features = prepare_features(pd.read_csv(DATA_DIR / "extended_features.csv"))
    validation = pd.read_csv(DATA_DIR / "Validation.csv")
    validation_patient_ids = validation["patient_id"].unique()

    feature_variants = {"base": baseline_features}
    feature_variants.update(build_feature_variants(extended_features))

    for model_name, model_filename, output_filename in MODEL_CONFIGS:
        model_array = load_model_array(model_filename)
        export_model_predictions(
            model_name,
            model_array,
            feature_variants[model_name],
            validation_patient_ids,
            output_filename,
        )


if __name__ == "__main__":
    main()
