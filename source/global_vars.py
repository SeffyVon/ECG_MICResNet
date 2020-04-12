#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:49:01 2020

@author: mac
"""

headers = ['Age', 'Sex', 'Dx']  
lead_feature_names = ['mean_RR','mean_Peaks','median_RR','median_Peaks','std_RR',
                'std_Peaks','var_RR','var_Peaks','skew_RR','skew_Peaks','kurt_RR',
                'kurt_Peaks']
leads = ['I','II','III',
		'aVR','aVL','aVF',
		'V1','V2','V3',
        'V4','V5','V6']
for ll in leads:
	for lfn in lead_feature_names:
		headers.append('{}_{}'.format(ll, lfn))

labels = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE']