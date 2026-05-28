from sklearn import metrics
def testing(modelarray,test_group,features):
    predictarray = [None] * len(modelarray)
    test=features[features["patient_id"].isin(test_group)]
    x = test.loc[:, 'asymmetry_score':'border_score']
    z = test["diagnostic"]
    y = []
    for item in z:
        y.append(item)
    predictions = [None] * len(test)
    for i in range(len(modelarray)):
        predictarray[i] = modelarray[i].predict(x)
    predict1, predict2, predict3, predict4, predict5 = predictarray
    for i in range(len(predict1)):
        val = 0
        temp = int(predict1[i]) + int(predict2[i]) + int(predict3[i]) + int(predict4[i]) + int(predict5[i])
        if temp > 2:
            val = 1
        predictions[i] = val
    acc = metrics.accuracy_score(y, predictions)
    AUC = metrics.roc_auc_score(y, predictions)
    return acc, AUC, predictions, y