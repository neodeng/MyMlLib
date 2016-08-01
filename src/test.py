#!/usr/bin/env python
#coding=utf-8

import matplotlib.pyplot as plt
from linear_regression import LinearRegression as lr

f=open('e0.txt','r')
x=[]
y=[]

for line in f.readlines():
    s=line.split('\t')
    s=[float(ele) for ele in s]
    x.append(s[:-1])
    y.append([s[-1]])

'''
print 'norm equation method:'
ns = lr('norm',False)
ns.fit(x,y)

print 'SGD method:'
ls = lr('SGD',False)
ls.fit(x,y)
'''
ls = lr('LOCAL',False)
yy=ls.fit(x,y)



#plot
X=[ele[1] for ele in x]
Y=[ele[0] for ele in y]


fig = plt.figure()
ax = fig.add_subplot(211)
ax.scatter(X,Y)
bx = fig.add_subplot(212)
bx.scatter(X,yy)
plt.show()



