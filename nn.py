import pandas as pd 
import numpy as np
import json

import plotly
import plotly.graph_objs as go

def model(df, feats_list, county):
    print(county)
    target = df[df['pos']==county][feats_list]
    distances = []
    for index, row in df[feats_list].iterrows():
        dist = 0
        for f in feats_list:
            dist += (float(target[f]) - float(row[f]))**2
        distances.append(np.sqrt(dist))
    df['distances'] = distances
    neighs = df[df['pos'] != county].sort_values('distances').iloc[:5]['pos'].to_list()
    return neighs 

    