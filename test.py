# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input
import scipy
from blackScholes import BlackScholes
from functionsAndFormulas import*

N=3 # time disrectization
S0=1 # initial value of the asset
strike=1 # strike for the call option 
T=1.0 # maturity
sigma=0.2 # volatility in Black Scholes
R=10 # number of Trajectories
Ktrain = 2
m=1



assetTest = BlackScholes(T, sigma,S0, N)
priceBS2 = assetTest.blackScholesPriceCall(strike)
paths = assetTest.generatePaths(Ktrain) 

def BS(S0, strike, T, sigma):
    return S0*scipy.stats.norm.cdf((np.log(S0/strike)+0.5*T*sigma**2)/(np.sqrt(T)*sigma))-strike*scipy.stats.norm.cdf((np.log(S0/strike)-0.5*T*sigma**2)/(np.sqrt(T)*sigma))

priceBS=BS(S0,strike,T,sigma)
#print(priceBS)
#print(priceBS2)


y = [np.random.normal(-(sigma)**2*T/(2*N),sigma*np.sqrt(T)/np.sqrt(N),(Ktrain,m)) for i in range(N)]
x = np.random.normal(-(sigma)**2*T/(2*N),sigma*np.sqrt(T)/np.sqrt(N),(Ktrain,m))

incr = increments(paths)

#print(y)
print("------------------------")
print(incr)
print("------------------------")
z = np.hsplit(incr, N)
print(z)
print(z[0].shape)











