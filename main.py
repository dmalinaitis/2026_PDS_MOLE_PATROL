"""
The Mole Patrol - train or load the eight Random Forest models and evaluate them.

    python main.py           evaluate the saved models in results/models/ (reproduces the report)
    python main.py --train   retrain all models from scratch, overwriting results/models/

Each model is an ensemble of 5 Random Forests, one per cross-validation fold. The script
prints each model's accuracy and AUC on the held-out validation patients and saves its
validation predictions to results/predictions/. When training, it also prints the
cross-validated AUC (mean +/- sd of each forest on its own fold's test patients).
"""
import pickle
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, roc_auc_score

from src.model import cancer, cross_validate, drop_colours, split, train, validate

# Model name -> (feature table, colour groups removed from it, saved model file).
# The base model uses the baseline table (original images). All other models use the
# extended table (hair/pen removed, RGB + greyscale added) and only differ in which colour
# spaces they keep, to answer the open question "which colour features work best?".
MODELS = {
    "base": ("baseline", [], "basemodel.pkl"),
    "extended": ("extended", ["rgb", "grey"], "extendedmodel.pkl"),  # HSV only
    "grey": ("extended", ["hsv", "rgb"], "greymodel.pkl"),
    "rgb": ("extended", ["hsv", "grey"], "rgbmodel.pkl"),
    "rgb_grey": ("extended", ["hsv"], "rgbgreymodel.pkl"),
    "hsv_grey": ("extended", ["rgb"], "hsvgreymodel.pkl"),
    "rgb_hsv": ("extended", ["grey"], "rgbhsvmodel.pkl"),
    "all": ("extended", [], "allcolormodel.pkl"),  # HSV + RGB + greyscale
}


def main(load_model=True):
    # Hold out the validation patients and split the rest into 5 patient-grouped folds
    train_groups, test_groups, val_patients = split(
        pd.read_csv("data/metadata.csv"), pd.read_csv("data/Validation.csv")
    )

    # Load both feature tables and turn the six diagnoses into the binary cancer label
    tables = {}
    for table in ("baseline", "extended"):
        tables[table] = pd.read_csv(f"data/{table}_features.csv")
        tables[table]["diagnostic"] = tables[table]["diagnostic"].apply(cancer)

    # Other scikit-learn versions split the folds differently, so retraining changes the models
    if not load_model and sklearn.__version__ != "1.7.2":
        print(f"Warning: the saved models were trained with scikit-learn 1.7.2, you have {sklearn.__version__}, "
              "so the retrained models will not reproduce the reported scores.\n")

    print(f"{'model':<10}{'val. accuracy':>14}{'val. AUC':>10}" + ("" if load_model else "   CV AUC"))
    for name, (table, removed, model_file) in MODELS.items():
        features = drop_colours(tables[table], removed)
        model_path = f"results/models/{model_file}"
        if load_model:
            with open(model_path, "rb") as f:
                forests = pickle.load(f)
        else:
            forests = train(features, train_groups)
            with open(model_path, "wb") as f:
                pickle.dump(forests, f)

        predictions = validate(forests, features, val_patients)
        predictions.to_csv(f"results/predictions/predictions_{name}.csv", index=False)

        # AUC is computed from the hard 0/1 votes (not probabilities), as in the report
        accuracy = accuracy_score(predictions["true_label"], predictions["predicted_label"])
        auc = roc_auc_score(predictions["true_label"], predictions["predicted_label"])
        line = f"{name:<10}{accuracy:>14.4f}{auc:>10.4f}"
        if not load_model:  # the fold test sets only match forests trained on this run's split
            cv_aucs = cross_validate(forests, features, test_groups)
            line += f"   {np.mean(cv_aucs):.3f} +/- {np.std(cv_aucs):.3f}"
        print(line)


if __name__ == "__main__":
    main(load_model="--train" not in sys.argv)
