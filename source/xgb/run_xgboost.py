import pandas as pd
import numpy as np
from xgb.global_vars import headers, labels

def run_xgboost(features,classes,models):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    df=pd.DataFrame(features.reshape((1,-1)), 
                    columns=headers)

    df_X = df.drop(['Dx'], axis=1)

    for i, ll in enumerate(labels): 
        current_label[i] = models[ll].predict(df_X)
        current_score[i] = models[ll].predict_proba(df_X)[0][0]

    return current_label, current_score