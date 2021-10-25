# -*- coding: utf-8 -*-

#import tensorflow as tf
#import numpy as np
#import pandas as pd
import numpy as np
from scipy.stats import norm
from blackScholes import BlackScholes
from matplotlib import pyplot as plt 
from functionsAndFormulas import*

r = 0
sigma = 0.2
T = 30/365
n = 30
numberOfPaths = 100000
S0 = 100
K = 100
alpha = 0.5
times = np.arange(0,T,T/n)

assetTest = BlackScholes(T, sigma,S0, n, r)

pathsTest = assetTest.generatePaths(numberOfPaths) 

callTest = call(K, pathsTest)

hedgeTest = assetTest.deltaHedgeCall(pathsTest ,K)

tradingErfolg = assetTest.tradingReturn(hedgeTest,pathsTest)

price = assetTest.blackScholesPriceCall(K)

loss =-callTest+price+ tradingErfolg

var = VaR(alpha, loss)
es = cvar(alpha, loss)


plt.hist(loss, bins = np.arange(-5,5,0.5)) 
plt.title(f'var {alpha} ') 
plt.show()


#print(pathsTest)
print(callTest)
print()
#print(tradingErfolg)
#print(hedgeTest)
print(loss)
#print(tradingErfolg-callTest)
print()
print(loss.std())
print()



