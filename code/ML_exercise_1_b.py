#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:45:47 2018

@author: air
"""

import pandas
import numpy
import os
import data as cd
import regression as cr
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

def main():
    ## Read Data
    trainingData = pandas.read_csv(os.path.join(cd.data_dir(), cd.DataSets.EX1B.value, 'train.csv'),
                                   header=0, index_col=0)
    yCols = ['y']
    xCols = trainingData.drop(columns=yCols).columns
    X = trainingData.loc[:, xCols]
    y = trainingData.loc[:, yCols]
    n, m = X.shape
    f2=pandas.DataFrame(numpy.square(X),index=X.index)
    f3=pandas.DataFrame(numpy.exp(X),index=X.index)
    f4=pandas.DataFrame(numpy.cos(X),index=X.index)
    f5=pandas.DataFrame(numpy.ones((n, 1)),index=X.index)
    F = pandas.concat([X,f2,f3,f4], axis=1)    

    FStd = F.std()
    standF = (F) / FStd

    F = pandas.concat([standF, f5], axis=1)

    reg = LassoCV(fit_intercept=False, cv=10, alphas=None)
    lassocv = reg.fit(F.as_matrix(), y.as_matrix().flatten())
    alpha = lassocv.alpha_ 
    W = lassocv.coef_
    pred = (W * FStd.append(pandas.Series(0))).dot(F.T)
    rmse = numpy.sqrt(mean_squared_error(y_true=y, y_pred=pred))
    print("RMSE %.6f" %rmse)
        
    weights = pandas.Series(W * FStd.append(pandas.Series(0)), index=F.columns)
    print("Final weights \n %s" %weights)
    
    weights.to_csv(path=os.path.join(cd.data_dir(), cd.DataSets.EX1B.value, 'sample_lasso_standardize.csv'), index=False)
   
if __name__ == '__main__':
    main()

