# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:05:12 2017

@author: Kyle
"""

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
#using numpy
#file = open('.\\datasets\\abalone.txt','r')
#data=file.read()
#print(data)
#file.close()

#using panda
f = '.\\datasets\\abalone.txt'
df = pd.read_csv(f,sep=',',names=['sex','length','diameter',
                                  'height','whole weight','shucked weight',
                                  'viscera weight','shell weight','rings'] )

#can convert to numpy array
#narray = df.values

regr = linear_model.LinearRegression()
