from sklearn import metrics
def testing(model,test_group,features):
    test_group = test_group.reset_index(drop=True)
    test=features[features["patient_id"].isin(test_group)]
    x = test.loc[:, 'asymmetry_score':'value_95p']
    y = test["diagnostic"]
    predictions = model.predict(x)
    acc = metrics.accuracy_score(y, predictions)
    AUC = metrics.roc_auc_score(y, predictions)
    return acc, AUC

