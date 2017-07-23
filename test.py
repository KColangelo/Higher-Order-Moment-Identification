# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 16:20:54 2017

@author: Kyle
"""



from __future__ import division
import numpy as np
from scipy.special import psi
from statsmodels.sandbox.regression.gmm import GMM



class GMMGamma(GMM):

    def __init__(self, *args, **kwds):
        # set appropriate counts for moment conditions and parameters
        # TODO: clean up signature
        kwds.setdefault('k_moms', 4)
        kwds.setdefault('k_params', 2)
        super(GMMGamma, self).__init__(*args, **kwds)


    def momcond(self, params):
        p0, p1 = params
        endog = self.endog
        error1 = endog - p0 / p1
        error2 = endog**2 - (p0 + 1) * p0 / p1**2
        error3 = 1 / endog - p1 / (p0 - 1)
        error4 = np.log(endog) + np.log(p1) - psi(p0)
        g = np.column_stack((error1, error2, error3, error4))
        return g

y = np.array([20.5, 31.5, 47.7, 26.2, 44.0, 8.28, 30.8, 17.2, 19.9, 9.96, 55.8, 25.2, 29.0, 85.5, 15.1, 28.5, 21.4, 17.7, 6.42, 84.9])



nobs = y.shape[0]
x = np.ones((nobs, 4))

model = GMMGamma(y, x, None)
beta0 = np.array([2, 0.1])
res = model.fit(beta0, maxiter=2, optim_method='nm', wargs=dict(centered=False))
print(res.summary())

