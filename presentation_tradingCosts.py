import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Concatenate, Dropout, Subtract, \
                        Flatten, MaxPooling2D, Multiply, Lambda, Add, Dot
from tensorflow.keras.backend import constant
from tensorflow.keras import optimizers

from tensorflow.keras import losses

#from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras import initializers
from tensorflow.keras.constraints import max_norm
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import scipy.stats as scipy
from scipy.stats import norm

# Trajectories of the Black scholes model
# Let it run to initialize the following parameters, the trajectories 
# are not needed afterwards

N=20 # time disrectization
S0=1 # initial value of the asset
strike=1 # strike for the call option 
T=1.0 # maturity
sigma=0.2 # volatility in Black Scholes
R=10 # number of Trajectories

#Definition of neural networks for heding strategies

m = 1 # dimension of price
d = 2 # number of layers in strategy
n = 32  # nodes in the first but last layers

def BS(S0, strike, T, sigma):
    return S0*scipy.norm.cdf((np.log(S0/strike)+0.5*T*sigma**2)/(np.sqrt(T)*sigma))-strike*scipy.norm.cdf((np.log(S0/strike)-0.5*T*sigma**2)/(np.sqrt(T)*sigma))

priceBS=BS(S0,strike,T,sigma)

m = 1 # dimension of price
d = 3 # number of layers in strategy
n = 8  # nodes hidden layers

# architecture is the same for all networks
layers = []
for j in range(N):
    for i in range(d):
        if i < d-1:
            nodes = n
            layer = Dense(nodes, activation='tanh',trainable=True,
                      kernel_initializer=initializers.RandomNormal(0,1),#kernel_initializer='random_normal',
                      bias_initializer='random_normal',
                      name=str(i)+str(j))
        else:
            nodes = m
            layer = Dense(nodes, activation='linear', trainable=True,
                          kernel_initializer=initializers.RandomNormal(0,0.1),#kernel_initializer='random_normal',
                          bias_initializer='random_normal',
                          name=str(i)+str(j))
        layers = layers + [layer]

#Implementing the loss function
# Inputs is the training set below, containing the price S0, 
#the initial hedging being 0, and the increments of the log price process 
price = Input(shape=(m,))
hedge = Input(shape=(m,))
hedgeeval = Input(shape=(m,))
premium = Input(shape=(m,))
costs = Input(shape=(m,))

tradingcost = Input(shape=(m,))
position = Input(shape=(m,))
costpara = 0.01

inputs = [price]+[position]+[costs]+[hedge]+[hedgeeval]+[premium]
outputhelper=[]

premium = Dense(m, activation='linear', trainable=True,
                kernel_initializer=initializers.RandomNormal(0,1),#kernel_initializer='random_normal',
                bias_initializer=initializers.RandomNormal(0,1))(premium)

for j in range(N):
    strategy = Concatenate()([price, position])
    strategyeval=Concatenate()([hedgeeval, position])
    for k in range(d):
        strategy= layers[k+(j)*d](strategy) # strategy at j is the hedging strategy at j , i.e. the neural network g_j
        strategyeval=layers[k+(j)*d](strategyeval)
    incr = Input(shape=(m,))
    logprice= Lambda(lambda x : K.log(x))(price)
    logprice = Add()([logprice, incr])
    pricenew=Lambda(lambda x : K.exp(x))(logprice)# creating the price at time j+1
    priceincr=Subtract()([pricenew, price])
    hedgenew = Multiply()([strategy, priceincr])
    
    change = Subtract()([strategy, position]) # this is deltat+1-deltat i.e. the change in our position of the asset
    absolutechange = Lambda(lambda x : K.abs(x))(change)
    costshelper = Multiply()([absolutechange,price])
    costshelper = Lambda(lambda x : 0.01*x)(costshelper)
    costs = Add()([costs,costshelper])
    
    
    #mult = Lambda(lambda x : K.sum(x,axis=1))(mult) # this is only used for m > 1
    hedge = Add()([hedge,hedgenew]) # building up the discretized stochastic integral
    inputs = inputs + [incr]
    outputhelper = outputhelper + [strategyeval]
    price=pricenew
    position = strategy # out current strategy will be the position in the next timestep
    
#after trading is done this is the cost of liquidating our position    
change = Subtract()([strategy, position]) # this is deltat+1-deltat i.e. the change in our position of the asset
absolutechange = Lambda(lambda x : K.abs(x))(change)
costhelper = Multiply()([absolutechange,price])
costshelper = Lambda(lambda x : 0.01*x)(costshelper)
costs = Add()([costs,costshelper])
    
payoff= Lambda(lambda x : 0.5*(K.abs(x-strike)+x-strike))(price) 
outputs = Subtract()([payoff,hedge]) 
outputs = Subtract()([outputs,premium]) # payoff minus price minus hedge 
outputs = Add()([outputs,costs])
outputs= [outputs] + outputhelper +[premium]
outputs = Concatenate()(outputs)

model_hedge_strat = Model(inputs=inputs, outputs=outputs)

Ktrain = 5*10**5
initialprice = S0

# xtrain consists of the price S0, 
#the initial hedging being 0, and the increments of the log price process 
xtrain = ([initialprice*np.ones((Ktrain,m))] +
          [np.zeros((Ktrain,m))]+
          [np.zeros((Ktrain,m))]+
          [np.zeros((Ktrain,m))]+
          [np.ones((Ktrain,m))] +
          [priceBS*np.ones((Ktrain,m))]+
          [np.random.normal(-(sigma)**2*T/(2*N),sigma*np.sqrt(T)/np.sqrt(N),(Ktrain,m)) for i in range(N)])

# why 1+N and not 1?
ytrain=np.zeros((Ktrain,1+N))

def custom_loss(y_true,y_pred):
    #return losses.mean_squared_error(y_true[0], y_pred[0])
    z = y_pred[:,0]-y_true[:,0]
    z=K.mean(K.square(z))
    return z

model_hedge_strat.compile(optimizer='adam',loss=custom_loss)

model_hedge_strat.summary()

for i in range(5):
    model_hedge_strat.fit(x=xtrain,y=ytrain, epochs=1,verbose=True)
# =============================================================================
# plt.hist(model_hedge_strat.predict(xtrain)[:,0])
# plt.show()
# print(np.std(model_hedge_strat.predict(xtrain)[:,0]))
# print(np.mean(model_hedge_strat.predict(xtrain)[:,N+1]))
# =============================================================================

Ktest=2*10**4
xtest = ([initialprice*np.ones((Ktest,m))] +
         [np.zeros((Ktest,m))]+
         [np.zeros((Ktest,m))]+
          [np.zeros((Ktest,m))]+
          [np.linspace(0.5,1.5,Ktest)] +#change this if you go to higher dimensions
          [priceBS*np.ones((Ktest,m))]+
          [np.random.normal(-(sigma)**2*T/(2*N),sigma*np.sqrt(T)/np.sqrt(N),(Ktest,m)) for i in range(N)])

plt.hist(model_hedge_strat.predict(xtest)[:,0])
plt.show()
print(np.std(model_hedge_strat.predict(xtest)[:,0]))
print(np.mean(model_hedge_strat.predict(xtest)[:,N+1]))
print(model_hedge_strat.predict(xtest)[0,N])
print(model_hedge_strat.predict(xtest)[0,N+1])

logincrements = xtest[6:6+N]
hedge = np.zeros(Ktest)
price = S0*np.ones((Ktest,N))
for k in range(N-1):
    helper = logincrements[k][:,]
    helper = helper.transpose()
    price[:,k+1] = price[:,k]*np.exp(helper[:])
    hedge[:] = hedge[:] + scipy.norm.cdf((np.log(price[:,k]/strike)+0.5*(T-k*T/N)*sigma**2)/(np.sqrt(T-k*T/N)*sigma))*(price[:,k+1]-price[:,k])
hedge[:]= hedge[:]-0.5*(np.abs(price[:,N-1]-strike)+(price[:,N-1]-strike))+priceBS
plt.hist(hedge)
plt.show()
print(np.std(hedge))
print(np.mean(hedge))




