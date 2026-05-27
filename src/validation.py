from sklearn import metrics
def testing(modelarray,test_group,features):
    predictarray = [None] * len(modelarray)
    test_group = test_group.reset_index(drop=True)
    test=features[features["patient_id"].isin(test_group)]
    x = test.loc[:, 'asymmetry_score':'border_score']
    y = test["diagnostic"]
    predictions = [None] * len(test)
    for i in range(len(modelarray)):
        predictarray[i] = modelarray[i].predict(x)
    predict1, predict2, predict3, predict4, predict5 = predictarray
    for i in range(len(predict1)):
        val = 0
        temp = predict1[i] + predict2[i] + predict3[i] + predict4[i] + predict5[i]
        if temp > 2:
            val = 1
        predictions[i] = val
    acc = metrics.accuracy_score(y, predictions)
    AUC = metrics.roc_auc_score(y, predictions)
    return acc, AUC, predictions, y