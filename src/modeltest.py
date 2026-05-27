from sklearn import metrics
def testing(model,test_group,features,i):
    test_group = test_group.reset_index(drop=True)
    test=features[features["patient_id"].isin(test_group)]
    x = test.loc[:, 'asymmetry_score':'border_score']
    y = test["diagnostic"]
    predictions = model.predict(x)
    acc = metrics.accuracy_score(y, predictions)
    AUC = metrics.roc_auc_score(y, predictions)
    return f"The accuracy of {model} {i} is {acc}, it's AUC is {AUC}."

