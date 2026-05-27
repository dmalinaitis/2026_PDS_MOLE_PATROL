from sklearn.ensemble import RandomForestClassifier
def treebase(basefeatures, train_group):
    basemodelarray = [None] * len(train_group)
    for i in range(len(train_group)):
        basemodelarray[i]=submodel(train_group[i], basefeatures)
    return basemodelarray


def submodel(train, basefeatures):
    clf = RandomForestClassifier(n_estimators=200,max_depth=4, random_state=11037)
    modeltraining=basefeatures[basefeatures["patient_id"].isin(train)]
    modeltraining=modeltraining.reset_index(drop=True)
    y = modeltraining["diagnostic"]
    x = modeltraining.loc[:, 'asymmetry_score':'border_score']
    model = clf.fit(x,y)
    return model