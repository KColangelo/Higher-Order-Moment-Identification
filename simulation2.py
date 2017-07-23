# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 17:11:43 2017

@author: Kyle
"""


from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import statsmodels.formula.api as sm
from statsmodels.sandbox.regression.gmm import GMM
from scipy.special import psi

class GMMa(GMM):
    def __init__(self, *args, **kwds):
        # set appropriate counts for moment conditions and parameters
        # TODO: clean up signature
        kwds.setdefault('k_moms', 3)
        kwds.setdefault('k_params', 2)
        super(GMMa, self).__init__(*args, **kwds)


    def momcond(self, params):
        b= params
        b=b.reshape(len(b),1)
        print(b)
        y = self.endog
        x = self.exog
        y = y.reshape(len(y),1)
        m1 = np.dot(((x[:,0]-2)**2).T,np.subtract(y,np.dot(x,b)))
        m2 = np.dot(((x[:,0]-2)**4).T,np.subtract(y,np.dot(x,b)))
        m3 = np.dot((x[:,1]-2).T,np.subtract(y,np.dot(x,b)))
        g = np.column_stack((m1,m2,m3))
        return g




beta1 = .9
beta2=.5
mu_x = 2
sigma_x2=1
sigma_epsilon2=.1
rho=0.5

means = [mu_x,mu_x,0]
n=10000
Omega = np.array([[sigma_x2,0,rho],[0,sigma_x2,0],[rho,0,sigma_epsilon2]])
betas = np.zeros((7,len(samples)))
k=100

#for i in range(1,k):
x1, x2, epsilon = np.random.multivariate_normal(means, Omega,n).T
y=x1*beta1 +x2*beta2 +epsilon

x1=pd.Series(x1)
x2=pd.Series(x2)
y=pd.Series(y)
df = pd.concat([y,x1,x2],axis=1)
df.columns = ['y','x1','x2']
x = pd.concat([x1,x2],axis=1)


x = np.array(x)   
y = np.array(y).reshape(n,1) 
model2 = GMMa(y, x, None)
beta0 = np.array([.9,.5]) 
res = model2.fit(beta0, maxiter=2)
print(res.summary())
    

















