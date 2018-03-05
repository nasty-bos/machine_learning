import numpy
from scipy import linalg


#######################################################################
def linear_regression(X, y, intercept=None):

    return {'done': False}

#######################################################################
def ridge_regression(X, y, lambdaParam, intercept=None):
    '''
    computes the analytical solution to the ridge regression problem

    Args:
        :param X:
        :param y:
        :param lambdaParam:
        :param intercept:

    :return:
    '''

    n, m = X.shape
    return linalg.inv(X.T.dot(X) + lambdaParam * numpy.eye(m)).dot(X.T.dot(y))