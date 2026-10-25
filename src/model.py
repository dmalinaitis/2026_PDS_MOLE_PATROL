"""
Modelling: everything that happens after feature extraction.

1. Labels     - the six PAD-UFES-20 diagnoses become a binary cancer label.
2. Features   - colour groups are removed from the extended table to build the model variants.
3. Split      - the validation patients are held out and the rest is split into 5 folds.
4. Training   - one Random Forest per fold, so every model is an ensemble of 5 forests.
5. Evaluation - each forest is scored on its own fold's test patients (cross-validation),
                then the 5 forests vote on the held-out validation images (final result).
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SEED = 11037  # random_state of both the fold split and the forests
CANCER = ["SCC", "BCC", "MEL"]  # cancer diagnoses; ACK, NEV and SEK are not cancer

# First and last column of each colour group in extended_features.csv
COLOUR_GROUPS = {
    "hsv": ("hue_mean", "value_95p"),
    "rgb": ("red_mean", "blue_95p"),
    "grey": ("grey_mean", "grey_95p"),
}


def cancer(diagnosis):
    """Binary label: '1' for a cancer diagnosis, '0' otherwise.

    The labels are strings because the saved models were trained on string labels.
    """
    return "1" if diagnosis in CANCER else "0"


def drop_colours(features, groups):
    """Return the feature table without the given colour groups ('hsv', 'rgb' and/or 'grey')."""
    for group in groups:
        first, last = COLOUR_GROUPS[group]
        features = features.drop(columns=features.loc[:, first:last].columns)
    return features


def model_inputs(features):
    """The columns the forests use: every column from asymmetry_score to border_score.

    In extended_features.csv the colour columns sit between those two, so they are included.
    In baseline_features.csv border_score comes directly after asymmetry_score (the HSV
    columns follow it), so the base model only uses the two shape features.
    """
    return features.loc[:, "asymmetry_score":"border_score"]


def split(metadata, validation):
    """Hold out the validation patients and split the remaining images into 5 folds.

    StratifiedGroupKFold keeps all images of a patient in the same fold (no patient
    leakage) while keeping the six diagnoses balanced across the folds.
    Returns the training patients and test patients of each fold, and the validation patients.
    """
    val_patients = validation["patient_id"].to_list()
    remaining = metadata[~metadata["patient_id"].isin(val_patients)].reset_index(drop=True)
    patients = remaining["patient_id"]

    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_groups, test_groups = [], []
    for train_idx, test_idx in folds.split(remaining["img_id"], remaining["diagnostic"], patients):
        train_groups.append(patients[train_idx])
        test_groups.append(patients[test_idx])
    return train_groups, test_groups, val_patients


def train(features, train_groups):
    """Fit one Random Forest per fold, each on the images of that fold's training patients."""
    forests = []
    for patients in train_groups:
        rows = features[features["patient_id"].isin(patients)].reset_index(drop=True)
        forest = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=SEED)
        forests.append(forest.fit(model_inputs(rows), rows["diagnostic"]))
    return forests


def cross_validate(forests, features, test_groups):
    """AUC of each forest on the test patients of its own fold."""
    aucs = []
    for forest, patients in zip(forests, test_groups):
        rows = features[features["patient_id"].isin(patients)]
        predictions = forest.predict(model_inputs(rows)).astype(int)
        aucs.append(roc_auc_score(rows["diagnostic"].astype(int), predictions))
    return aucs


def validate(forests, features, patients):
    """Majority vote of the 5 forests on every image of the given (validation) patients.

    An image is predicted as cancer when at least 3 of the 5 forests say so.
    Returns one row per image: img_id, patient_id, true_label and predicted_label.
    """
    rows = features[features["patient_id"].isin(patients)].reset_index(drop=True)
    votes = sum(forest.predict(model_inputs(rows)).astype(int) for forest in forests)
    return pd.DataFrame({
        "img_id": rows["img_id"],
        "patient_id": rows["patient_id"],
        "true_label": rows["diagnostic"].astype(int),
        "predicted_label": (votes > 2).astype(int),
    })
