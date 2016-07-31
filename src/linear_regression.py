#!/usr/bin/env python
#-*- coding: UTF-8 -*-

"""
    Several Linear Regression Methods including:

    Norm Equations:  beta=(X^(T)X)^(-1)X^(T)y

    Gradient Method:  1) batch gradient decent(BGD): 
                         for each parameter use all the examples to upgrade
                      2) stochastic gradient decent(SGD):
                         upgrade all the parameters for each example
                     
    Locally Weighted:
    Shrinkage Method:
    Forward Regression:
"""

# @neodeng
# 7/27/2016  First edition
# License: BSD 3 clause

import numpy as np

class LinearRegression:
    """ 
        LinearRegression Model:
        X(n,m) array of features:
               n is number of examples;
               m is the number of features of each example
        y(n,1) array of results   
    """
    
    def __init__(self, method='norm', fit_intercept=True):
        self.method = method
        self.fit_intercept = fit_intercept
        self.coef_ = []
        
    def fit(self, X, y):
        X = np.mat(X)
        y = np.mat(y)
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0],1)), X]
        if self.method == 'norm':
            self.coef_ = LinearRegression.fit_norm(X, y)
            print self.coef_
            return
        if self.method == 'SGD':
            self.coef_ = LinearRegression.fit_sgd(X, y, alpha=0.01, iternum=100)
            print self.coef_
            return
        if self.method == 'LOCAL':
            self.coef_ = LinearRegression.fit_local(X, y)
            print self.coef_
            return

    @staticmethod
    def fit_norm(X, y):
        """ The norm equation method """
        try:
            coef = (((X.T)*X).I)*(X.T)*y
            return coef
        except LinAlgError, e:
            print 'Cant not use norm equation method!', e
            return
    
    @staticmethod
    def fit_sgd(X, y, alpha, iternum):
        """ The stochastic gradient method 
            alpha | the learning rate
            iternum | the number of iteration
        """
        n = X.shape[0]  # number of examples
        m = X.shape[1]  # number of features (parameters)
        coef = np.zeros(m)  # initialize of parameters vector
        coef = np.mat(coef).T
        for j in range(iternum):
            for i in range(n):
                error = y[i] - X[i] * coef
                error = error.getA()[0][0]
                coef = coef + alpha * error * X[i].T
        return coef
                
            
            
    
                

