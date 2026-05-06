from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import os
def cancer(x):
  '''Function to change class labels to binary value of C and NC. '''
  ctype = ['SCC','BCC','MEL']
  if x in ctype:
    return('C')
  else:
    return('NC')
def spliting():
    '''Function that uses the metadata file of a given dataset, takes the image and patient ids with the lesion type and uses them 
    make a cross validation split using stratified group k folding.'''
    currentdir = os.getcwd()
    os.chdir(os.path.dirname(currentdir))
    #changing working directory to ensure easy access to metadata for split
    df = pd.read_csv('./data/metadata.csv')
    valdf = pd.read_csv('./data/Validation.csv')
    #validation handleing, due to bin sorting problem this had to be done by hand.
    valpat = valdf['patient_id']
    valpatlist = valpat.to_list()
    df = df[~df['patient_id'].isin(valpatlist)]
    group = df['patient_id']
    y = df['diagnostic']
    x = df['img_id']
    #reading the metadata into a data frame and then isolating the patient_id, image_id and class of skin lesion.
    split = 5
    #split defining
    testindexarray = [None] * split
    testgrouparray = [None] * split
    trainindexarray = [None] * split
    traingrouparray = [None] * split
    # array initialization
    sg = StratifiedGroupKFold(n_splits=split,shuffle=True,random_state=11037)
    for i, (train_index, test_index) in enumerate(sg.split(x, y, group)):
        trainindexarray[i] = train_index
        traingrouparray[i] = group[train_index]
        testindexarray[i] = test_index
        testgrouparray[i] = group[test_index]
    return trainindexarray, traingrouparray, testindexarray, testgrouparray, valpatlist
