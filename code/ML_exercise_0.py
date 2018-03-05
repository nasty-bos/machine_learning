#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 19:32:03 2018

@author: Anastasiya
"""

import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load the datasets
dir = "/Users/air/Documents/Master/2. Semester/intro to machine learning/code/machine_learning/data/"

trainSet = pd.read_csv(os.path.join(dir, 'train.csv'), header=0, index_col=0, float_precision='round_trip')
testSet = pd.read_csv(os.path.join(dir, 'test.csv'), header=0, index_col=0, float_precision='round_trip') 
sampleSet = pd.read_csv(os.path.join(dir, 'sample.csv'), header=0, index_col=0, float_precision='round_trip')

linreg = LinearRegression(fit_intercept=False)

xColumns = ['x' + str(i+1) for i in range(10)]
yColumns = ['y']

linreg.fit(trainSet.loc[:, xColumns], trainSet.loc[:, yColumns])
print(linreg.intercept_)
print(linreg.coef_)
y_pred=linreg.predict(testSet)
print(y_pred)

print(np.sqrt(metrics.mean_squared_error(testSet.loc[:, xColumns].mean(axis=1), y_pred)))

# write to pandas Series object 
yPred = pd.DataFrame(y_pred, index=testSet.index, columns=['y'])
yPred.to_csv(os.path.join(dir, 'sample.csv'))


# check 
assert np.sqrt(metrics.mean_squared_error(testSet.loc[:, xColumns].mean(axis=1), y_pred)) < 1e-6,\
    'regression does not have perfect fit for known functional form'