#!/usr/bin/env python
#coding: UTF-8

"""
Several Linear Regression Methods including:
Norm Equations:  beta=(X^(T)X)^(-1)X^(T)y
Gradient Method:
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
        X(n,m) arrary of features:
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
        Y = np.mat(y)
        if self.method == 'norm':
            if self.fit_intercept:
                X = np.c_(ones(X.ndim,1), X)
            self.coef_ = LinearRegression.fit_norm(X,Y)
            print self.coef_
            return

    @staticmethod
    def fit_norm(X, y):
        """ The norm equation method """
        try:
            coef = (((X.T)*X).I)*(X.T)*Y
            return coef
        except LinAlgError, e:
            print 'Cant not use norm equation method!', e
            return
            
    
                

