import pandas as pd
import pickle
import src.traintestsplit
import src.basemodeltrain
import src.cancer
import src.modeltest
import src.validation
import src.drop

def main(extendedfeatures_path,basemodel_path,extendedmodel_path,greymodel_path,rgbmodel_path,featuresbaseline_path,validation_path,allcolormodel_path,rgb_hsvmodel_path,hsv_greymodel_path,rgb_greymodel_path,prediction_results_base_path,prediction_results_extended_path,prediction_results_grey_path,prediction_results_rgb_path,prediction_results_rgb_grey_path,prediction_results_rgb_hsv_path,prediction_results_hsv_grey_path,prediction_results_all_path,load_model):
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
    train_index, train_group, test_index, test_group, validation = src.traintestsplit.spliting(df,valdf)

    if load_model:
        # load the model
        basemodelarray = pickle.load(open(basemodel_path, "rb"))
        extendedmodelarray = pickle.load(open(extendedmodel_path, "rb"))
        greymodelarray = pickle.load(open(greymodel_path, "rb"))
        rgbmodelarray = pickle.load(open(rgbmodel_path, "rb"))
        rgb_greymodelarray = pickle.load(open(rgb_greymodel_path, "rb"))
        hsv_greymodelarray = pickle.load(open(hsv_greymodel_path, "rb"))
        rgb_hsvmodelarray = pickle.load(open(rgb_hsvmodel_path, "rb"))
        allcolormodelarray = pickle.load(open(allcolormodel_path, "rb"))
        pass
    else:
        # train the classifier (using logistic regression as an example)
        basefeatures = pd.read_csv(featuresbaseline_path)
        extendedfeatures = pd.read_csv(extendedfeatures_path)
        basefeatures["diagnostic"] = basefeatures["diagnostic"].apply(src.cancer.cancer)
        extendedfeatures["diagnostic"] = extendedfeatures["diagnostic"].apply(src.cancer.cancer)
        rgb_greyfeatures = src.drop.hsv(extendedfeatures)
        greyfeatures = src.drop.rgb(rgb_greyfeatures)
        rgb_hsvfeatures = src.drop.grey(extendedfeatures)
        hsvfeatures = src.drop.rgb(rgb_hsvfeatures)
        rgbfeatures = src.drop.grey(rgb_greyfeatures)
        hsv_greyfeatures = src.drop.rgb(extendedfeatures)

        basemodelarray = src.basemodeltrain.treebase(basefeatures, train_group)
        extendedmodelarray = src.basemodeltrain.treebase(hsvfeatures, train_group)
        greymodelarray = src.basemodeltrain.treebase(greyfeatures, train_group)
        rgbmodelarray = src.basemodeltrain.treebase(rgbfeatures, train_group)
        rgb_greymodelarray = src.basemodeltrain.treebase(rgb_greyfeatures, train_group)
        hsv_greymodelarray = src.basemodeltrain.treebase(hsv_greyfeatures, train_group)
        rgb_hsvmodelarray = src.basemodeltrain.treebase(rgb_hsvfeatures, train_group)
        allcolormodelarray = src.basemodeltrain.treebase(extendedfeatures, train_group)
        #gini based tree model, depth tbd. models to be trained off of groups from splitting.
        #
        # model1 = clf.fit(features, class), model2, etc each based on pulling from train_group[i] for features from set
        #for testing each model looks at features and predices classifier, return prediction majority
        for i in range(len(basemodelarray)):
            src.modeltest.testing(basemodelarray[i], test_group[i], basefeatures,i)
            src.modeltest.testing(extendedmodelarray[i], test_group[i], hsvfeatures,i)
            src.modeltest.testing(greymodelarray[i], test_group[i], greyfeatures,i)
            src.modeltest.testing(rgbmodelarray[i], test_group[i], rgbfeatures,i)
            src.modeltest.testing(rgb_greymodelarray[i], test_group[i], rgb_greyfeatures,i)
            src.modeltest.testing(hsv_greymodelarray[i], test_group[i], hsv_greyfeatures,i)
            src.modeltest.testing(rgb_hsvmodelarray[i], test_group[i], rgb_hsvfeatures,i)
            src.modeltest.testing(allcolormodelarray[i], test_group[i], extendedfeatures,i)
        #

        # save the model.
        pickle.dump(basemodelarray, open(basemodel_path, "wb"))
        pickle.dump(extendedmodelarray, open(extendedmodel_path, "wb"))
        pickle.dump(greymodelarray, open(greymodel_path, "wb"))
        pickle.dump(rgbmodelarray, open(rgbmodel_path, "wb"))
        pickle.dump(rgb_greymodelarray, open(rgb_greymodel_path, "wb"))
        pickle.dump(hsv_greymodelarray, open(hsv_greymodel_path, "wb"))
        pickle.dump(rgb_hsvmodelarray, open(rgb_hsvmodel_path, "wb"))
        pickle.dump(allcolormodelarray, open(allcolormodel_path, "wb"))
        #save via pickle. pickle.dump(model once trained)
        pass

    # test the classifier.
    # NO NOT UNCOMMENT OR TEST THIS FUNCTION UNTIL ALL OTHER TASKS ARE DONE, FAILURE TO DO SO WILL COMPROMISE PROJECT.
    base_accuracy, base_AUC, base_predictions, true_class = src.validation.testing(basemodelarray, validation, basefeatures)
    extended_accuracy, extended_AUC, extended_predictions, true_class = src.validation.testing(extendedmodelarray, validation, hsvfeatures)
    grey_accuracy, grey_AUC, grey_predictions, true_class = src.validation.testing(greymodelarray, validation, greyfeatures)
    rgb_accuracy, rgb_AUC, rgb_predictions, true_class = src.validation.testing(rgbmodelarray, validation, rgbfeatures)
    rgb_grey_accuracy, rgb_grey_AUC, rgb_grey_predictions, true_class = src.validation.testing(rgb_greymodelarray, validation, rgb_greyfeatures)
    hsv_grey_accuracy, hsv_grey_AUC, hsv_grey_predictions, true_class = src.validation.testing(hsv_greymodelarray, validation, hsv_greyfeatures)
    rgb_hsv_accuracy, rgb_hsv_AUC, rgb_hsv_predictions, true_class = src.validation.testing(rgb_hsvmodelarray, validation, rgb_hsvfeatures)
    allcolor_accuracy, allcolor_AUC, allcolor_predictions, true_class = src.validation.testing(allcolormodelarray, validation, extendedfeatures)


    # write test results to CSV.



if __name__ == "__main__":
    metadata_path = "./data/metadata.csv"
    extendedfeatures_path = "./data/extended_features.csv"
    featuresbaseline_path = "./data/baseline_features.csv"
    validation_path = "./data/validation.csv"
    prediction_results_base_path = "./results/predictions/predictions_base.csv"
    prediction_results_extended_path = "./results/predictions/predictions_extended.csv"
    prediction_results_grey_path = "./results/predictions/predictions_grey.csv"
    prediction_results_rgb_path = "./results/predictions/predictions_rgb.csv"
    prediction_results_rgb_grey_path = "./results/predictions/predictions_rgb_grey.csv"
    prediction_results_rgb_hsv_path = "./results/predictions/predictions_rgb_hsv.csv"
    prediction_results_hsv_grey_path = "./results/predictions/predictions_hsv_grey.csv"
    prediction_results_all_path = "./results/predictions/predictions_all.csv"
    basemodel_path = "./results/models/basemodel.pkl"
    extendedmodel_path = "./results/models/extendedmodel.pkl"
    greymodel_path = "./results/models/greymodel.pkl"
    rgbmodel_path = "./results/models/rgbmodel.pkl"
    rgb_greymodel_path = "./results/models/rgbgreymodel.pkl"
    hsv_greymodel_path = "./results/models/hsvgreymodel.pkl"
    rgb_hsvmodel_path = "./results/models/rgbhsvmodel.pkl"
    allcolormodel_path = "./results/models/allcolormodel.pkl"


    load_model = False

    main(extendedfeatures_path,basemodel_path,extendedmodel_path,greymodel_path,rgbmodel_path,featuresbaseline_path,validation_path,allcolormodel_path,rgb_hsvmodel_path,hsv_greymodel_path,rgb_greymodel_path,prediction_results_base_path,prediction_results_extended_path,prediction_results_grey_path,prediction_results_rgb_path,prediction_results_rgb_grey_path,prediction_results_rgb_hsv_path,prediction_results_hsv_grey_path,prediction_results_all_path,load_model)