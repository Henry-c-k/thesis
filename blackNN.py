import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scipy
from scipy.stats import norm
import tensorflow as tf
from blackScholes import BlackScholes
from functionsAndFormulas import*


# tensorboard stuff
#import os
#import time
#from tensorflow.keras.callbacks import TensorBoard
#NAME = "blackNN-{}".format(int(time.time()))
#tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
#model_name = f'BlackScholesNN'
#model_log_dir = os.path.join(logsdir, f'model_{model_name}')
#tb_callback = TensorBoard(log_dir=logsdir)
#TensorBoard(log_dir=model_log_dir, histogram_freq=0,profile_batch=0,write_graph=0)
#logsdir = os.path.abspath("C:/Users/Henry/Desktop/masterarbeit/deepHedging/coding/master/tensorboard")
#if  not os.path.exists(logsdir):
#    os.mkdir(logsdir)

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
import tensorflow.keras.backend as K

N=20 # time disrectization
S0=1 # initial value of the asset
strike=1 # strike for the call option 
T=1.0 # maturity
sigma=0.2 # volatility in Black Scholes
R=10 # number of Trajectories
initialprice = S0
Ktrain = 2*10**4
Ktest  = 2*10**4

#
assetTest = BlackScholes(T, sigma,S0, N)
priceBS = assetTest.blackScholesPriceCall(strike)
pathsTrain = assetTest.generatePaths(Ktrain) 
pathsTest = assetTest.generatePaths(Ktest)
incrementsTrain = increments(pathsTrain)
incrementsTest = increments(pathsTest)
assetInputTrain = listInput(incrementsTrain, N)
assetInputTest = listInput(incrementsTest, N)

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
premium = Input(shape=(m,))

inputs = [price]+[hedge]+[premium]

zzz = Dense(m, activation='linear', trainable=True,
                kernel_initializer=initializers.RandomNormal(0,1),
                bias_initializer=initializers.RandomNormal(0,1))

premium = zzz(premium)

for j in range(N):
    strategy = price
    for k in range(d):
        strategy= layers[k+(j)*d](strategy) # strategy at j is the hedging strategy at j , i.e. the neural network g_j
    incr = Input(shape=(m,))
    pricenew=Add()([price, incr])
    hedgenew = Multiply()([strategy, incr])
    #mult = Lambda(lambda x : K.sum(x,axis=1))(mult) # this is only used for m > 1
    hedge = Add()([hedge,hedgenew]) 
    inputs = inputs + [incr]
    price=pricenew
 
payoff= Lambda(lambda x : 0.5*(K.abs(x-strike)+x-strike))(price) 
outputs = Subtract()([payoff,hedge]) 
outputs = Subtract()([outputs,premium]) # payoff minus price minus hedge 
outputs= [outputs] +[premium]
outputs = Concatenate()(outputs)

model_hedge_strat = Model(inputs=inputs, outputs=outputs)


# xtrain consists of the price S0, 
#the initial hedging being 0, and the increments of the log price process 
# =============================================================================
# xtrain = ([initialprice*np.ones((Ktrain,m))] +
#           [np.zeros((Ktrain,m))]+
#           [priceBS*np.ones((Ktrain,m))]+
#           [np.random.normal(-(sigma)**2*T/(2*N),sigma*np.sqrt(T)/np.sqrt(N),(Ktrain,m)) for i in range(N)])
# =============================================================================

xtrain = ([initialprice*np.ones((Ktrain,m))] +
          [np.zeros((Ktrain,m))]+
          [priceBS*np.ones((Ktrain,m))]+
          assetInputTrain)

ytrain=np.zeros((Ktrain,1))

def custom_loss(y_true,y_pred):
    #return losses.mean_squared_error(y_true[0], y_pred[0])
    z = y_pred[:,0]-y_true[:,0]
    z=K.mean(K.square(z))
    return z

#entropic risk measure as loss
# =============================================================================
# def custom_loss(y_true,y_pred):
#      #return losses.mean_squared_error(y_true[0], y_pred[0])
#      z = y_pred[:,0]#-y_true[:,0]
#      z = K.exp(z)
#      z = K.mean(z)
#      z = K.log(z)
#      return z
# =============================================================================

model_hedge_strat.compile(optimizer='adam',loss=custom_loss)


#model_hedge_strat.summary()
# =============================================================================
# for i in range(5):
#     model_hedge_strat.fit(x=xtrain,y=ytrain,epochs=1,verbose=True)
# =============================================================================
model_hedge_strat.fit(
    x=xtrain,
    y=ytrain,
    epochs=5,
    verbose=True#,callbacks=[tensorboard]
    )

Ktest=2*10**4
xtest = ([initialprice*np.ones((Ktest,m))] +
          [np.zeros((Ktest,m))]+
          [priceBS*np.ones((Ktest,m))]+
          assetInputTest)

plt.hist(model_hedge_strat.predict(xtest)[:,0])
plt.show()
print(np.std(model_hedge_strat.predict(xtest)[:,0]))
print(np.mean(model_hedge_strat.predict(xtest)[:,1]))

# =============================================================================
# logincrements = xtest[5:5+N]
# hedge = np.zeros(Ktest)
# price = S0*np.ones((Ktest,N))
# for k in range(N-1):
#     helper = logincrements[k][:,]
#     helper = helper.transpose()
#     price[:,k+1] = price[:,k]*np.exp(helper[:])
#     hedge[:] = hedge[:] + scipy.norm.cdf((np.log(price[:,k]/strike)+0.5*(T-k*T/N)*sigma**2)/(np.sqrt(T-k*T/N)*sigma))*(price[:,k+1]-price[:,k])
# hedge[:]= hedge[:]-0.5*(np.abs(price[:,N-1]-strike)+(price[:,N-1]-strike))+priceBS
# plt.hist(hedge)
# plt.show()
# print(np.std(hedge))
# print(np.mean(hedge))
# 
# =============================================================================








