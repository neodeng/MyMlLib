#!/usr/bin/env python
#coding=utf-8

from linear_regression import LinearRegression as lr

f=open('e0.txt','r')
x=[]
y=[]

for line in f.readlines():
    s=line.split('\t')
    s=[float(ele) for ele in s]
    x.append(s[:-1])
    y.append([s[-1]])

print 'norm equation method:'
ns = lr('norm',False)
ns.fit(x,y)

print 'SGD method:'
ls = lr('SGD',False)
ls.fit(x,y)
