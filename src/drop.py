import pandas as pd
def rgb(x):
    x=x.drop(x.loc[:,'red_mean':'blue_95p'], axis=1)
    return x

def grey(x):
    x=x.drop(x.loc[:,'grey_mean':'grey_95p'], axis=1)
    return x

def hsv(x):
    x=x.drop(x.loc[:,'hue_mean':'value_95p'], axis=1)
    return x