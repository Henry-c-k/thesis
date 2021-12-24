# -*- coding: utf-8 -*-

#import tensorflow as tf
#import numpy as np
#import pandas as pd
import numpy as np
from scipy.stats import norm
from blackScholes import BlackScholes
from matplotlib import pyplot as plt 
from functionsAndFormulas import*


import tensorflow.keras as tfk
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Lambda, Add, Subtract, Multiply
from tensorflow.keras import initializers


#model and asset parameters
# =============================================================================
# r = 0
# sigma = 0.04
# T = 30/365
# n = 30
# numberOfPaths = 10
# S0 = 100
# K = 100
# alpha = 0.5
# times = np.arange(0,T,T/n)
# =============================================================================

N=20
S0=1
strike=1
T=1.0
sigma=0.2
R=10
r=0

#create the assets, generate sample paths and apply the option to the paths
assetTest = BlackScholes(T, sigma,S0, N, r)
priceBS = assetTest.blackScholesPriceCall(strike)
pathsTest = assetTest.generatePaths(R) 
callTest = call(strike, pathsTest)

print("The call option is valued at " + str(priceBS))

for i in range(R):
    plt.plot(pathsTest[i,:])
plt.show()

m=1  #dimension of price
d=2  #number of layers in strategy
n=32 #nodes in hidden layer 

layers = []
for j in range(N):
    for i in range(d):
        if i<d-1:
            nodes = n
            layer  = Dense(nodes, activation='tanh', trainable=True,
                           kernel_initializer=initializers.RandomNormal(0,1),
                           bias_initializer='random_normal',
                           name=str(i)+str(j))
        else:
            nodes = m
            layer  = Dense(nodes, activation='linear', trainable=True,
                           kernel_initializer=initializers.RandomNormal(0,1),
                           bias_initializer='random_normal',
                           name=str(i)+str(j))
        layers = layers + [layer]

price = Input(shape=(m,))
hedge = Input(shape=(m,))
inputs = [price] + [hedge]

for j in range(N):
    strategy = price
    for k in range(d):
        strategy= layers[k+(j)*d](strategy) # strategy at j is the hedging strategy at j , i.e. the neural network g_j
    incr = Input(shape=(m,))
    logprice= Lambda(lambda x : K.log(x))(price)
    logprice = Add()([logprice, incr])
    pricenew=Lambda(lambda x : K.exp(x))(logprice)# creating the price at time j+1
    priceincr=Subtract()([pricenew, price])
    hedgenew = Multiply()([strategy, priceincr])
    #mult = Lambda(lambda x : K.sum(x,axis=1))(mult) # this is only used for m > 1
    hedge = Add()([hedge,hedgenew]) # building up the discretized stochastic integral
    inputs = inputs + [incr]
    price=pricenew
payoff= Lambda(lambda x : 0.5*(K.abs(x-strike)+x-strike) - priceBS)(price) 
outputs = Subtract()([payoff,hedge]) # payoff minus priceBS minus hedge

inputs = inputs
outputs= outputs

model_hedge = Model(inputs=inputs, outputs=outputs)

model_hedge.summary()

Ktrain = 2*10**4
initialprice = S0

# xtrain consists of the price S0, 
#the initial hedging being 0, and the increments of the log price process 
xtrain = ([initialprice*np.ones((Ktrain,m))] +
          [np.zeros((Ktrain,m))]+
          [np.random.normal(-(sigma)**2*T/(2*N),sigma*np.sqrt(T)/np.sqrt(N),(Ktrain,m)) for i in range(N)])

ytrain=np.zeros((Ktrain,1))

#sgd=optimizers.SGD(lr=0.0001)
#model_hedge.compile(optimizer=sgd,loss='mean_squared_error')
model_hedge.compile(optimizer='adam',loss='mean_squared_error')

for i in range(50):
    model_hedge.fit(x=xtrain,y=ytrain, epochs=1,verbose=True)
    #plt.hist(model_hedge.predict(xtrain))
    #plt.show()
    #print(np.mean(model_hedge.predict(xtrain)))

Ltest = 10**4
        
xtest=([initialprice*np.ones((Ktrain,m))] +
          [np.zeros((Ktrain,m))]+
          [np.random.normal(-(sigma)**2*T/(2*N),(sigma)*np.sqrt(T)/np.sqrt(N),(Ktrain,m)) for i in range(N)])
plt.hist(model_hedge.predict(xtest))
plt.show()
print(np.std(model_hedge.predict(xtest)))
print(np.mean(model_hedge.predict(xtest)))






















#hedgeTest = assetTest.deltaHedgeCall(pathsTest ,K)
#tradingErfolg = assetTest.tradingReturn(hedgeTest,pathsTest)
#loss =-callTest+price+ tradingErfolg
#var = VaR(alpha, loss)
#es = cvar(alpha, loss)

#plots and graphs
# =============================================================================
# for i in range(10):
#     plt.plot(pathsTest[i,:])
# plt.show()
# 
# for i in range(10):
#     plt.plot(hedgeTest[i,:])
# plt.show()
# 
# plt.hist(loss, bins = np.arange(-5,5,0.5)) 
# plt.title(f'var {alpha} ') 
# plt.show()
# =============================================================================





