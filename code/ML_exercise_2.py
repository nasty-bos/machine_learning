#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 20:04:54 2018

@author: air
"""


import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import data as cd
from sklearn import datasets, metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load the datasets
trainSet = pd.read_csv(os.path.join(cd.data_dir(), cd.DataSets.EX2.value, 'train.csv'), header=0, index_col=0, float_precision='round_trip')
testSet = pd.read_csv(os.path.join(cd.data_dir(), cd.DataSets.EX2.value, 'test.csv'), header=0, index_col=0, float_precision='round_trip')
sampleSet = pd.read_csv(os.path.join(cd.data_dir(), cd.DataSets.EX2.value, 'sample.csv'), header=0, index_col=0, float_precision='round_trip')

# Now apply the transformations to the data:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data

prcptrn = Perceptron(fit_intercept=False)

xColumns = ['x' + str(i+1) for i in range(16)]
yColumns = ['y']

scaler.fit(trainSet.loc[:, xColumns])
scaledTrainSet = scaler.transform(trainSet.loc[:, xColumns])
scaledTestSet = scaler.transform(testSet.loc[:, xColumns])

#prcptrn.fit(scaledTrainSet, trainSet.loc[:, yColumns].as_matrix().flatten())

#y_pred=prcptrn.predict(scaledTestSet)
#print(y_pred)
from sklearn.neural_network import MLPClassifier
mlpclass = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5), random_state=1, max_iter=1e3)

# grid-search for optimum paramters 
paramters = {'alpha': 10.0 ** -np.arange(1, 7, 0.1)}
optMPL = GridSearchCV(mlpclass, paramters)
optMPL.fit(scaledTrainSet, trainSet.loc[:, yColumns].as_matrix().flatten())

#  MLP TRAINING
mlpclass.fit(scaledTrainSet, trainSet.loc[:, yColumns].as_matrix().flatten())
optMPL.fit(scaledTrainSet, trainSet.loc[:, yColumns].as_matrix().flatten())

# prediction in-sample
y_in_sample = mlpclass.predict(scaledTrainSet)
yOpt_in_sample = optMPL.predict(scaledTrainSet)

# classification statistics
dim = len(yOpt_in_sample)
inSampleConfusionMatrix = confusion_matrix(trainSet.loc[:, yColumns], yOpt_in_sample)
accuracy = np.sum(np.diag(inSampleConfusionMatrix)) / dim
print("Using optimal parameter alpha, model accuracy %.4f " %accuracy)

# prediction
y_pred = mlpclass.predict(scaledTestSet)

# write to pandas Series object 
yPred = pd.DataFrame(y_pred, index=testSet.index, columns=['y'])
yPred.to_csv(os.path.join(cd.data_dir(), cd.DataSets.EX2.value, 'sample.csv'))
