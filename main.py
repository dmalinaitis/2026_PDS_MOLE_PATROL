import pandas as pd
import pickle

def main(features_path, prediction_results_path, model_path, load_model):
    """
    Docstring for main
    
    :param features_path: Path to the features csv used as input to the model (e.g. ./data/features.csv).
    :param prediction_results_path: Path to save the output predictions of the model (e.g. ./result/predictions/predictions_MODEL.csv).
    :param model_path: Path to save or load the trained model (e.g. ./result/predictions/predictions_MODEL.csv).
    :param load_model: Boolean to train the model and save it to model_path if False, load it from model_path if True. 
    """
    
    # load dataset and validation CSV file
    df=pd.read_csv(metadata_path)
    valdf=pd.read_csv(validation_path)

    # split the dataset into training and testing sets.
    from src import traintestsplit
    train_index, train_group, test_index, test_group, validation = traintestsplit.spliting(df,valdf)

    if load_model:
        # load the model
        pickle.load("saved model here.")
        pass
    else:
        # train the classifier (using logistic regression as an example)
        clf = tree.DecisionTreeClassifier(max_depth=3, random_state=11037)
        #gini based tree model, depth tbd. models to be trained off of groups from splitting.
        #
        # model1 = clf.fit(features, class), model2, etc each based on pulling from train_group[i] for features from set
        #for testing each model looks at features and predices classifier, return prediction majority
        #

        # save the model.
        #save via pickle. pickle.dump(model once trained)
        pass

    # test the classifier.


    # write test results to CSV.



if __name__ == "__main__":
    metadata_path = "./data/metadata.csv"
    features_path = "./data/features.csv"
    validation_path = "./data/validation.csv"
    prediction_results_path = "./result/predictions/predictions_MODEL.csv"
    model_path = "./result/predictions/predictions_MODEL.csv"
    load_model = False

    main(features_path, prediction_results_path,model_path,load_model)