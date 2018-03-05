import pandas
import numpy
import os
import code.data as cd
import code.regression as cr


def main():

    ## Read Data
    trainingData = pandas.read_csv(os.path.join(cd.data_dir(), cd.DataSets.EX1.value, 'train.csv'),
                                   header=0, index_col=0)
    yCols = ['y']
    xCols = trainingData.drop(columns=yCols).columns

    ## Set-up cross validation
    lambdaParam = [.1, 1, 10, 100, 1000]
    rmseVec = pandas.Series(index=[str(i) for i in lambdaParam], name='RMSE')
    N = 50
    rem = 0
    for l in lambdaParam:
        measuredRMSE = []
        for k in range(10):
            if k==9:
                rem = 1
            fold = numpy.arange(k * (N), ((k + 1) * 50) - 1 + rem)
            mask = trainingData.index.isin(fold)

            X_t = trainingData.loc[~mask, xCols]
            y_t = trainingData.loc[~mask, yCols]

            X = trainingData.loc[mask, xCols]
            y = trainingData.loc[mask, yCols]

            B = cr.ridge_regression(X=X_t, y=y_t, lambdaParam=l)
            betas = pandas.Series(data=B.flatten(), index=xCols)
            yFit = pandas.Series(X.dot(betas).values, index=y.index, name='yFit')
            measuredRMSE.append(numpy.sqrt(numpy.mean((y.iloc[:, 0] - yFit)**2)))

        rmseVec[str(l)] = numpy.mean(measuredRMSE)

    print(rmseVec)
    rmseVec.to_csv(os.path.join(cd.data_dir(), cd.DataSets.EX1.value, '__sample.csv'), index=False)

if __name__ == '__main__':
    main()





