#!/usr/bin/env python
#-*- coding: UTF-8 -*-

"""
    Several Linear Regression Methods including:

    Norm Equations:  beta=(X^(T)X)^(-1)X^(T)y

    Gradient Method:  1) batch gradient decent(BGD): 
                         for each parameter use all the examples to upgrade
                      2) stochastic gradient decent(SGD):
                         upgrade all the parameters for each example
                     
    Locally Weighted:  W = w(i,i) = exp(-(x(i)-x)^2/(2k^2))
                       x is the query point, k is the bandwidth parameter
                       
    Shrinkage Method:  Ridge Regression -> L2 Regulation
                       beta for min{||X*beta-y||^2 + lambda*||beta||^2}
                       beta_hat = (X.T*X+lambda*I)^-1*X.T*y
                       
    Forward Regression:  first. z-score regularize for X and y
                         second. iterate iternum:
                                     minErr = inf
                                     for each Beta[j]:
                                         +|- step
                                         minErr = min(minErr, Err)
                                         beta_max
                                     beta = beta_max
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
        self.yHat = []
        
    def fit(self, X, y):
        X = np.mat(X)
        y = np.mat(y)
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0],1)), X]
        if self.method == 'norm':
            self.coef_ = LinearRegression.fit_norm(X, y)
            return self.coef_
        if self.method == 'sgd':
            self.coef_ = LinearRegression.fit_sgd(X, y, alpha=0.01, iternum=100)
            return self.coef_
        if self.method == 'local':
            self.yHat = LinearRegression.fit_local(X, y, k=0.01)
            return self.yHat
        if self.method == 'ridge':
            self.coef_ = LinearRegression.fit_ridge(X, y, lmd=0.001)
            return self.coef_
        if self.method == 'stage':
            coef_list = LinearRegression.fit_stage(X, y, step=0.01, iternum=1000) 
            return coef_list
        

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
    def fit_sgd(X, y, alpha=0.01, iternum=100):
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
                
    @staticmethod
    def fit_local(X, y, k=0.01):
        """ The local weighted regression method
            k | the bandwidth parameter
        """
        n = X.shape[0]
        m = X.shape[1]
        yHat = []
        for j in range(n):
            W = np.mat(np.eye((n)))
            for i in range(n):
                diff = X[j,:] - X[i,:]
                W[i,i] = np.exp(-diff*diff.T/2/k/k)
            coef = ((X.T*W*X).I)*X.T*W*y
            yH = X[j,:]*coef
            yHat.append(yH[0,0])
        return yHat
            
    @staticmethod
    def fit_ridge(X, y, lmd=0.01):
        """ The ridge regression method
            lambda | the penalize parameter
        """
        m = X.shape[1]
        E = np.mat(np.eye(m))
        try:
            coef = (X.T*X+lmd*E).I*X.T*y
            return coef
        except LinAlgError, e:
            print 'Cant not use norm equation method!', e
            return
            
    @staticmethod
    def fit_stage(X, y, step=0.01, iternum=100):
        """ The stage wise method
            step | change of beta[j]
            iternum | the number of iteration
        """
        #X = LinearRegression._regularize(X)
        #y = y - np.mean(y)
        n, m = X.shape
        W = np.zeros((iternum,m))
        w = np.mat(np.zeros((m,1)))
        wtest = w.copy()
        wmax = w.copy()
        for i in range(iternum):
            minErr = np.inf
            for j in range(m):
                for sig in [-1, 1]:
                    wtest = w.copy()
                    wtest[j] += step*sig
                    ytest = X*wtest
                    rssE = LinearRegression._rss(ytest, y)
                    if rssE < minErr:
                        minErr = rssE
                        wmax = wtest
            w = wmax.copy()
            W[i,:] = w.T
        return W
        
    @staticmethod
    def _regularize(A):
        m = A.shape[1]
        for i in range(m):
            mean = np.mean(A[:,i])
            std = np.std(A[:,i])
            A[:,i] = (A[:,i] - mean) / std
        return A
        
    @staticmethod
    def _rss(ytest, y):
        yerr = y - ytest
        rss = yerr.T * yerr
        return rss[0,0]
        
        

