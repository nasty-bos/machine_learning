import pandas
import numpy
import code.data as cd


#######################################################################
def linear_regression(X, y, intercept=None):

    return {'done': False}

#######################################################################
def ridge_cross_validation(X, y, lambdaParam, intercept=None):
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
    return numpy.linalg.inv(X.T.dot(X) + lambdaParam * numpy.eye(m)).dot(X.T.dot(y))