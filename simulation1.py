# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:48:33 2017

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
        b,c = params
        endog = self.endog
        exog = self.exog
        endog = endog.reshape(len(exog),1)
        m1 = np.dot((exog**2).T,np.subtract(endog,exog*b))
        m2 = np.dot((exog**4).T,np.subtract(endog,exog*b))
        m3 = c
        print(c)
        g = np.column_stack((m1, m2, m3))
        return g




beta = .9
mu_x = 2
sigma_x2=1
sigma_epsilon2=.1
rho=.3

means = [mu_x,0]
samples = [1000,10000,100000]
Omega = np.array([[sigma_x2,rho],[rho,sigma_epsilon2]])
betas = np.zeros((7,len(samples)))
count = 0
for i in samples:
    x, epsilon = np.random.multivariate_normal(means, Omega,i).T
    z = pd.Series(np.random.normal(3,1,i).T)
    y=x*beta +epsilon
    
    x=pd.Series(x)
    y=pd.Series(y)
    df = pd.concat([y,x],axis=1)
    df.columns = ['y','x']
    
    model = sm.ols(formula='y~x',data=df).fit()   
    betas[0,count] = model.params[1]
    
    w1 = (x-2)**2
    betas[1,count] = 1/(np.dot(w1,x))*np.dot(w1,y)
    
    w2 = (x-2)**4
    betas[2,count]= 1/(np.dot(w2,x))*np.dot(w2,y)
    
    w3 = (x-2)**6
    betas[3,count] = 1/(np.dot(w3,x))*np.dot(w3,y)
    
    w4 = (x-2)**8
    betas[4,count] = 1/(np.dot(w4,x))*np.dot(w4,y)
    
    w5 = (x-2)**10
    betas[5,count] = 1/(np.dot(w5,x))*np.dot(w5,y)
    
    betas[6,count] = 1/(np.dot(z,x))*np.dot(z,y)
    count = count+1
    
    model2 = GMMa(np.array(y).reshape(i,1), np.array(x).reshape(i,1), None)
    beta0 = np.array([.9,.1])
    res = model2.fit(beta0, maxiter=2, optim_method='nm', wargs=dict(centered=False))
    print(res.summary())
    
plt.xscale('log')    
plt.plot(samples,betas[0,])    
plt.plot(samples,betas[1,])
plt.plot(samples,betas[2,])
plt.plot(samples,betas[3,])
plt.plot(samples,betas[4,])
plt.plot(samples,betas[5,])
plt.plot(samples,betas[6,])


















