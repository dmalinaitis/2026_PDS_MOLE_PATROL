from sklearn import tree
def treebase(basefeatures, train_group):
    basemodelarray = [None] * len(train_group)
    for i in range(len(train_group)):
        basemodelarray[i]=submodel(train_group[i], basefeatures)
    return basemodelarray


def submodel(train, basefeatures):
    clf = tree.DecisionTreeClassifier(max_depth=3, random_state=11037)
    modeltraining=basefeatures[basefeatures["patient_id"].isin(train)]
    modeltraining=modeltraining.reset_index(drop=True)
    y = modeltraining["diagonstic"]
    x = modeltraining.columns[2:]
    model = clf.fit(x,y)
    return model