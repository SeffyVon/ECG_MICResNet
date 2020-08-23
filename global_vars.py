#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:49:01 2020

@author: mac
"""
import numpy as np

headers = ['Age', 'Sex', 'Dx']  
lead_feature_names = ['age', 'sex', 'mean_RR','mean_Peaks','median_RR','median_Peaks','std_RR',
                'std_Peaks','var_RR','var_Peaks','skew_RR','skew_Peaks','kurt_RR', 'kurt_Peaks']
leads = ['I','II','III',
		'aVR','aVL','aVF',
		'V1','V2','V3',
        'V4','V5','V6']

normal_class = '426783006'
equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
lad_class = '39732003'
rad_class = '47665007'

for ll in leads:
	for lfn in lead_feature_names:
		headers.append('{}_{}'.format(ll, lfn))

#labels = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE']

import pandas as pd
weights_csv = pd.read_csv('weights.csv')
columns = [int(idx) for idx in weights_csv.keys()[1:]]
weights = weights_csv.iloc[:,1:].to_numpy()
# sorted_idx = np.argsort(columns)
# weights = weights_csv.iloc[sorted_idx,sorted_idx+1].to_numpy()
# columns = np.array(columns)[sorted_idx]

Dx_map = pd.read_csv('dx_mapping_scored.csv')
Dx_map_unscored = pd.read_csv('dx_mapping_unscored.csv')
labels = Dx_map['SNOMED CT Code'].to_numpy()

# equivalent_mapping
equivalent_mapping = {}
for class1, class2 in equivalent_classes:
    equivalent_mapping[class1] = class2
    
normal_idx = np.argwhere(labels==int(normal_class))


disable_tqdm = True
enable_writer = False
run_name_base = 'ECGBagResNet_trial0_MIL_5segs_3000len_pos2_fixed_balanced_fullset'
run_name = run_name_base
n_segments = 5
max_segment_len = 3000



