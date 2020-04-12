#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 20:39:44 2020

@author: mac
"""

import pickle
import numpy as np

def train_xgboost(X, y, model_name='default_xgb', params={}):
    """
    input: features (dX, dy)
    output: models
    """

    # Train model
    print('Training classifier...')
    model = train_xgboost_model(X, y, params)
    
    with open('xgb/saved/'+model_name+'.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    return model

# ========================================================================
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import fbeta_score

def score_wrapper(y_true, y_pred):
    print("score wrapper")
    return fbeta_score(y_pred, y_true, beta=2) 

class XGBClassifierNew(XGBClassifier):
    
    def __init__(self, 
                 max_depth=3, learning_rate=0.1, n_estimators=100,
                 verbosity=1, silent=None,
                 objective="binary:logistic", booster='gbtree',
                 n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0, seed=None, missing=None, 
                 **kwargs):
        
        super(XGBClassifierNew, self).__init__(
            max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
            verbosity=verbosity, silent=silent, objective=objective, booster=booster,
            n_jobs=n_jobs, nthread=nthread, gamma=gamma,
            min_child_weight=min_child_weight, max_delta_step=max_delta_step,
            subsample=subsample, colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel, colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight,
            base_score=base_score, random_state=random_state, seed=seed, missing=missing,
            **kwargs) #XGBClassifierNew, self
    
    

    def fit(self, X, y, **kwargs):
        print("fit")
        weights = np.ones((len(y),))
        weights[np.argwhere(y == 1).flatten()] = 10
        return super().fit(X, y, sample_weight=weights)
        
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # eval_set=[(X_train, y_train), (X_test, y_test)]
    
        # return super().fit(X=X_train, y=y_train, 
        #                    eval_metric=['aucpr'],
        #                    #eval_metric=score_wrapper, 
        #                    eval_set=eval_set, **kwargs)
        


from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
kf = KFold(5)
def train_xgboost_model(X, y, params):
    
    pipe = OneVsRestClassifier(XGBClassifierNew(**params))
    
    param_grid = {"estimator__C": uniform(0.7, 0.3),
              "estimator__gamma": uniform(0, 0.8),
             # "estimator__learning_rate": uniform(0.003, 0.3), # default 0.1 
              "estimator__max_depth": randint(4, 10), # default 3
              "estimator__n_estimators": randint(100, 1000), # default 100
              "estimator__subsample": uniform(0.6, 0.4),
              "estimator__k_neighbors": randint(5, 10)}

    clf = RandomizedSearchCV(pipe, param_grid, n_jobs=2,
                         random_state=42, n_iter=2, cv=kf, 
                         verbose=1, return_train_score=True)
    
    clf.fit(X=X, y=y)
    return clf

if __name__ == '__main__':
    params = {'booster':'gbtree', 
          'objective':'binary:logistic'}
    clf = XGBClassifierNew(**params)
    print(clf.get_params())
    
    
    