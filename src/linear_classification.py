#!/usr/bin/env python
#-*- coding: UTF-8 -*-

"""
    Several Linear Classification Methods:
    
    Logistic Regression: h(X) = 1/(1+exp(-beta.T*X)) 

"""

# @neodeng
# 8/8/2016 First Edition
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression(object):
    """
        LogisticRegression Methods:
        X(n,m) array of features
        y(n,1) array of classes
    """
    
    def __init__(self, method='sgd', fit_intercept=True):
        self.method = method
        self.fit_intercept = fit_intercept
        self.coef_ = []
        
    def fit(self, X, y):
        X = np.mat(X)
        y = np.mat(y)
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0],1)), X]
        if self.method == 'sgd':
            self.coef_ = LogisticRegression.fit_sgd(X, y, alpha=0.001, iternum=500)
            return self.coef_
            
    
    @staticmethod
    def fit_sgd(X, y, alpha=0.01, iternum=100):
        """ The stochastic gradient ascent (SGA) method"""
        n,m = X.shape
        coef = np.mat(np.zeros(m)).T
        for j in range(iternum):
            for i in range(n):
                y_hat = LogisticRegression._y_hat(coef, X[i])
                error = y[i] - y_hat
                error = error[0,0]
                coef = coef + alpha * error * X[i].T
        return coef
    
    @staticmethod
    def fit_newton(X, y, iternum=100):
        """ 
            The newton-raphson method for LR 
            beta: = beta - H.I * gradient(l)
            here H is the Hessian Matrix
        """
        n,m = X.shape
        coef = np.mat(np.zeros(m)).T
        for j in range(iternum):
            
                
        
        
    @staticmethod
    def _y_hat(beta, x):
        """ 
            The function that return the y hat
            beta(m,1) the weights vector
            x(1,m) one example
        """
        y_h = 1.0 / (1.0 + np.exp(-x * beta))
        return y_h
        
    @staticmethod
    def draw_2d(X, y, coef, axis):
        ax1,ax2 = axis
        x1 = []
        x2 = []
        y1 = []
        y2 = []
        n = len(X)     
        for i in range(n):
            if y[i] == 1:
                x1.append(X[i][ax1])
                y1.append(X[i][ax2])
            else:
                x2.append(X[i][ax1])
                y2.append(X[i][ax2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x1, y1, s=30, c='red', marker='s')
        ax.scatter(x2, y2, s=30, c='blue')
        x = [-3+0.1*i for i in range(61)]
        y = [(-coef[0,0]-coef[1,0]*ele)/coef[2,0] for ele in x]
        ax.plot(x,y)
        plt.show()        