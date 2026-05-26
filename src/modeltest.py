def testing(model,test_group,features):
    test_group = test_group.reset_index(drop=True)
    test=features[features["patient_id"].isin(test_group)]
    x = test.loc[:, 'asymmetry_score':'value_95p']
    y = test["diagnostic"]
    predictions = model.predict(x)
    predicted_and_true_class = []
    for item1, item2 in zip(predictions,y):
        predicted_and_true_class.append(predictions,y)
    return predicted_and_true_class

