#basic utility
import numpy as np
from scipy.stats import norm
import os

#tensorflow
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.keras.models import Model

#plotting
import matplotlib.pyplot as plt

#my own code
from deepHedgingModel import*
from deepHedgingModelPremium import*
from deepHedgingModelVolume import*

from merton import*
from heston import*
from bergomi import*
from blackScholes import*

from functionsAndFormulas import*
from callOption import*
from losses import*

"""
This is the collection of all hedging experiments done in order to not have 25 different scripts.
The file is separated into 
1. definition model / hedging parameters
2. generation of input data
3. creating models and training them
4. create and print results of heding experiments

to use: uncomment the respective loading of data, the model and then the output
"""



#parameters of the hedging task
N=30                    #number of time steps
S0=100.0               #starting price
T=30/365                   #final time
Tf = 45/365             #maturity of variance swap (not the same as T because of bergomi instability)
costType='proportional' #type of transaction costs
eps = 0.01             #transaction costs parameter (0.0 means no costs)
optionType='call'       #type of option (does nothing for now)
strike = S0             #strike price of the call option
K = strike

#parameters of the rBergomi model (the real market)
a = -0.4        #H-0.5 (Hurst param)
H = 0.1
v0Berg = 0.235**2   #xi
etaBerg = 1.9
rhoBerg = -0.9 

#Heston parameters (fitted to the rBergomi parameters)
lamHest = 3.9810393221890084   #alpha
vbar = 0.09327160815810764     #b
etaHest = 1.704203041025321    #sigma
rhoHest = -0.716492483305834
v0Hest = 0.05126161966697793 

#Merton parameters (fitted to the rBergomi parameters)
sigmaMert = 0.198
vMert = 0.048924488315176824
mMert = 0.0
lamMert = 2.0829758475877087

#Black Scholes parameter (fitted to analytic Heston price)
sigmaBS = 0.208345


#parameters of the machine learning model
Ktrain =500000     #number of asset paths to be simulated and then used in training
testRatio=0.2          #number of asset paths to be used in testing as fraction of Ktrain
numberOfSimulations=int(Ktrain*(1+testRatio))  #number of total simulations
testSize = int(Ktrain*testRatio)                #number of test simulations
m = 17                  #nodes hidden layers
d = 2                   #number of hidden layers in strategy
n = 1                   #dimension of price
lr = 0.005#1e-2               #learning rate
batchSize = 256         #batch size
epochs = 1             #epochs
useBatchNorm=True      #status of batch normalization
weightInit = 'random_normal'     #how to initialize weights
biasInit = 'random_normal'
denseActivation = 'relu'          #activation from one layer to the next
outputActivation = 'sigmoid'            #activation of the output of each network
optimizer  = Adam(learning_rate=lr)     #using Adam as our optimizer

#additional stuff
lossParam=1.0
seed=0

################################################################################

# #creating assets from all 3 models
# assetsBerg = Bergomi(T, N, a, rhoBerg, etaBerg, v0Berg, S0, Tf)
# SBerg, swapBerg, lnSBerg, vBerg = assetsBerg.modelInput(numberOfSimulations)
# payoffBerg = callPayoff(K, SBerg)

# assetsHest = Heston(T, etaHest, lamHest, vbar, rhoHest, S0, N, v0Hest, Tf)
# lnSHest, vHest = assetsHest.generateHestonPaths(numberOfSimulations)
# SHest, swapHest = assetsHest.generateAssetPaths(lnSHest, vHest, numberOfSimulations)
# payoffHest = callPayoff(K, SHest)

# assetsMert = Merton(T, N, S0, lamMert, mMert, vMert, sigmaMert)
# SMert = assetsMert.generatePaths(numberOfSimulations)
# lnSMert, vMert = assetsMert.getInformation(SMert)
# swapMert = assetsMert.getSwap(vMert)
# payoffMert = callPayoff(K, SMert)

# assetsBS = BlackScholes(T, sigmaBS, S0, N)
# priceBS = assetsBS.blackScholesPriceCall(strike)
# pathsBS = assetsBS.generatePaths(numberOfSimulations)
# swapBS = np.full_like(pathsBS, sigmaBS*T)           
# payoffBS = callPayoff(strike, pathsBS)

# np.savez("./data/rBergomi/rBergomiInput", SBerg=SBerg, swapBerg=swapBerg, lnSBerg=lnSBerg, vBerg=vBerg, payoffBerg=payoffBerg)
# np.savez("./data/heston/hestonInput", SHest=SHest, swapHest=swapHest, lnSHest=lnSHest, vHest=vHest, payoffHest=payoffHest)
#np.savez("./data/merton/mertonInput", SMert=SMert, swapMert=swapMert, lnSMert=lnSMert, vMert=vMert, payoffMert=payoffMert)
#np.savez("./data/BS/BSInput", SBS=pathsBS, swapBS=swapBS, payoffBS=payoffBS)

rBergomiData = np.load("./data/rBergomi/rBergomiInput.npz")
SBerg = rBergomiData['SBerg']
swapBerg = rBergomiData['swapBerg']
lnSBerg=rBergomiData['lnSBerg']
vBerg=rBergomiData['vBerg']
payoffBerg=rBergomiData['payoffBerg']

hestonData = np.load("./data/heston/hestonInput.npz")
SHest = hestonData['SHest']
swapHest = hestonData['swapHest']
lnSHest=hestonData['lnSHest']
vHest=hestonData['vHest']
payoffHest=hestonData['payoffHest']

mertonData = np.load("./data/merton/mertonInput.npz")
SMert = mertonData['SMert']
swapMert = mertonData['swapMert']
lnSMert=mertonData['lnSMert']
vMert=mertonData['vMert']
payoffMert=mertonData['payoffMert']

# BSData = np.load("./data/BS/BSInput.npz")
# SBS = BSData['SBS']
# swapBS = BSData['swapBS']
# payoffBS=BSData['payoffBS']

################################################################################

priceAssetBerg = np.stack((SBerg),axis=1)
priceSwapBerg = np.stack((swapBerg),axis=1)
I1Berg = lnSBerg.transpose()
I2Berg = vBerg.transpose()
xAllBerg = createInput(priceAssetBerg, priceSwapBerg, I1Berg, I2Berg, payoffBerg, N)
[xtrainBerg, xtestBerg] = trainTestSplit(xAllBerg, testSize=testSize)

priceAssetHest = np.stack((SHest),axis=1)
priceSwapHest = np.stack((swapHest),axis=1)
I1Hest = lnSHest.transpose()
I2Hest = vHest.transpose()
xAllHest = createInput(priceAssetHest, priceSwapHest, I1Hest, I2Hest, payoffHest, N)
[xtrainHest, xtestHest] = trainTestSplit(xAllHest, testSize=testSize)

priceAssetMert = np.stack((SMert),axis=1)
priceSwapMert = np.stack((swapMert),axis=1)
I1Mert = lnSMert.transpose()
I2Mert = vMert.transpose()
xAllMert = createInput(priceAssetMert, priceSwapMert, I1Mert, I2Mert, payoffMert, N)
[xtrainMert, xtestMert] = trainTestSplit(xAllMert, testSize=testSize)

# priceAssetBS = np.stack((SBS),axis=1)
# priceSwapBS = np.stack((swapBS),axis=1)
# I1BS = np.log(priceAssetBS)
# I2BS = priceSwapBS/T
# xAllBS = createInput(priceAssetBS, priceSwapBS, I1BS, I2BS, payoffBS, N)
# [xtrainBS, xtestBS] = trainTestSplit(xAllBS, testSize=testSize)
# [Strain, Stest] = trainTestSplit([SBS], testSize=testSize)

################################################################################
#concatenate input data from all models
payoff3NN = np.hstack((payoffBerg,payoffHest,payoffMert))
price3NN = np.hstack((priceAssetBerg,priceAssetHest,priceAssetMert))
swap3NN = np.hstack((priceSwapBerg,priceSwapHest,priceSwapMert))
I13NN = np.hstack((I1Berg,I1Hest,I1Mert))
I23NN = np.hstack((I2Berg,I2Hest,I2Mert))
xAll3NN = createInput(price3NN, swap3NN, I13NN, I23NN, payoff3NN, N)
[xtrain3NN, xtest3NN] = trainTestSplit(xAll3NN, testSize=300000)

# payoff2NN = np.hstack((payoffBerg,payoffHest))
# price2NN = np.hstack((priceAssetBerg,priceAssetHest))
# swap2NN = np.hstack((priceSwapBerg,priceSwapHest))
# I12NN = np.hstack((I1Berg,I1Hest))
# I22NN = np.hstack((I2Berg,I2Hest))
# xAll2NN = createInput(price2NN, swap2NN, I12NN, I22NN, payoff2NN, N)
# [xtrain2NN, xtest2NN] = trainTestSplit(xAll2NN, testSize=200000)

################################################################################
#solo models

# modelBergomi = deepModel(N=N, m=m, eps=eps, useBatchNorm=useBatchNorm, weightInit=weightInit,
#                   biasInit=biasInit, denseActivation=denseActivation, 
#                   outputActivation=outputActivation)
# #lossBergomi = MSE(modelBergomi.output)
# lossBergomi = entropyLoss(modelBergomi.output,1.0)
# modelBergomi.add_loss(lossBergomi)
# modelBergomi.compile(optimizer=optimizer)
# modelBergomi.fit(x=xtrainBerg, batch_size=batchSize, epochs=epochs, \
#           validation_data=xtestBerg, verbose=1)
    
# modelHeston = deepModel(N=N, m=m, eps=eps, useBatchNorm=useBatchNorm, weightInit=weightInit,
#                   biasInit=biasInit, denseActivation=denseActivation, 
#                   outputActivation=outputActivation)
# #lossHeston = MSE(modelHeston.output)
# lossHeston = entropyLoss(modelHeston.output,1.0)
# modelHeston.add_loss(lossHeston)
# modelHeston.compile(optimizer=optimizer)
# modelHeston.fit(x=xtrainHest, batch_size=batchSize, epochs=epochs, \
#           validation_data=xtestHest, verbose=1)
 
# modelMerton = deepModel(N=N, m=m, eps=eps, useBatchNorm=useBatchNorm, weightInit=weightInit,
#                   biasInit=biasInit, denseActivation=denseActivation, 
#                   outputActivation=outputActivation)
# #lossMerton = MSE(modelMerton.output)
# lossMerton = entropyLoss(modelMerton.output,1.0)
# modelMerton.add_loss(lossMerton)
# modelMerton.compile(optimizer=optimizer)
# modelMerton.fit(x=xtrainMert, batch_size=batchSize, epochs=epochs, \
#           validation_data=xtestMert, verbose=1)    

################################################################################
#BS experiments

# modelBSpremium = deepModelPremium(N=N, m=m, eps=eps, useBatchNorm=useBatchNorm, weightInit=weightInit,
#                   biasInit=biasInit, denseActivation=denseActivation, 
#                   outputActivation=outputActivation)
# lossBSpremium = MSE(modelBSpremium.output[0])
# #lossBSpremiu, = entropyLoss(modelBSpremium.output[0],1.0)
# modelBSpremium.add_loss(lossBSpremium)
# modelBSpremium.compile(optimizer=optimizer)
# modelBSpremium.fit(x=xtrainBS, batch_size=batchSize, epochs=epochs, \
#           validation_data=xtestBS, verbose=1)   

# modelBS = deepModel(N=N, m=m, eps=eps, useBatchNorm=useBatchNorm, weightInit=weightInit,
#                   biasInit=biasInit, denseActivation=denseActivation, 
#                   outputActivation=outputActivation)
# #lossBS = MSE(modelBS.output)
# lossBS = entropyLoss(modelBS.output,1.0)
# modelBS.add_loss(lossBS)
# modelBS.compile(optimizer=optimizer)
# modelBS.fit(x=xtrainBS, batch_size=batchSize, epochs=epochs, \
#           validation_data=xtestBS, verbose=1)     

################################################################################
#joint models 

# model3NN = deepModel(N=N, m=m, eps=eps, useBatchNorm=useBatchNorm, weightInit=weightInit,
#                   biasInit=biasInit, denseActivation=denseActivation, 
#                   outputActivation=outputActivation)
# #loss3NN = MSE(model3NN.output)
# loss3NN = entropyLoss(model3NN.output,1.0)
# model3NN.add_loss(loss3NN)
# model3NN.compile(optimizer=optimizer)
# model3NN.fit(x=xtrain3NN, batch_size=batchSize, epochs=epochs, \
#           validation_data=xtest3NN, verbose=1) 

# model2NN = deepModel(N=N, m=m, eps=eps, useBatchNorm=useBatchNorm, weightInit=weightInit,
#                   biasInit=biasInit, denseActivation=denseActivation, 
#                   outputActivation=outputActivation)
# loss2NN = MSE(model2NN.output)
# model2NN.add_loss(loss2NN)
# model2NN.compile(optimizer=optimizer)
# model2NN.fit(x=xtrain2NN, batch_size=batchSize, epochs=epochs, \
#           validation_data=xtest2NN, verbose=1) 
 
################################################################################
#tracking trading volume   

# modelBergomi = deepModelVolume(N=N, m=m, eps=eps, useBatchNorm=useBatchNorm, weightInit=weightInit,
#                   biasInit=biasInit, denseActivation=denseActivation, 
#                   outputActivation=outputActivation)
# lossBergomi = MSE(modelBergomi.output[0])
# #lossBergomi = entropyLoss(modelBergomi.output[0],1.0)
# modelBergomi.add_loss(lossBergomi)
# modelBergomi.compile(optimizer=optimizer)
# modelBergomi.fit(x=xtrainBerg, batch_size=batchSize, epochs=epochs, \
#           validation_data=xtestBerg, verbose=1)
    
# modelHeston = deepModelVolume(N=N, m=m, eps=eps, useBatchNorm=useBatchNorm, weightInit=weightInit,
#                   biasInit=biasInit, denseActivation=denseActivation, 
#                   outputActivation=outputActivation)
# lossHeston = MSE(modelHeston.output[0])
# #lossHeston = entropyLoss(modelHeston.output[0],1.0)
# modelHeston.add_loss(lossHeston)
# modelHeston.compile(optimizer=optimizer)
# modelHeston.fit(x=xtrainHest, batch_size=batchSize, epochs=epochs, \
#           validation_data=xtestHest, verbose=1)
 
# modelMerton = deepModelVolume(N=N, m=m, eps=eps, useBatchNorm=useBatchNorm, weightInit=weightInit,
#                   biasInit=biasInit, denseActivation=denseActivation, 
#                   outputActivation=outputActivation)
# lossMerton = MSE(modelMerton.output[0])
# #lossMerton = entropyLoss(modelMerton.output[0],1.0)
# modelMerton.add_loss(lossMerton)
# modelMerton.compile(optimizer=optimizer)
# modelMerton.fit(x=xtrainMert, batch_size=batchSize, epochs=epochs, \
#           validation_data=xtestMert, verbose=1) 

model3NN = deepModelVolume(N=N, m=m, eps=0.0, useBatchNorm=useBatchNorm, weightInit=weightInit,
                  biasInit=biasInit, denseActivation=denseActivation, 
                  outputActivation=outputActivation)
loss3NN = MSE(model3NN.output[0])
#loss3NN = entropyLoss(model3NN.output,1.0)
model3NN.add_loss(loss3NN)
model3NN.compile(optimizer=optimizer)
model3NN.fit(x=xtrain3NN, batch_size=batchSize, epochs=epochs, \
          validation_data=xtest3NN, verbose=1)
    
model3NNeps = deepModelVolume(N=N, m=m, eps=0.01, useBatchNorm=useBatchNorm, weightInit=weightInit,
                  biasInit=biasInit, denseActivation=denseActivation, 
                  outputActivation=outputActivation)
loss3NN = MSE(model3NNeps.output[0])
#loss3NN = entropyLoss(model3NN.output,1.0)
model3NNeps.add_loss(loss3NN)
model3NNeps.compile(optimizer=optimizer)
model3NNeps.fit(x=xtrain3NN, batch_size=batchSize, epochs=epochs, \
          validation_data=xtest3NN, verbose=1)

################################################################################   
#reusing models if necessary (many of these are outdated and only saved during testin phase)
 
#modelBergomi.save("./models/rBergomiModelTransactionCosts")
#modelHeston.save("./models/HestonModelTransactionCosts")
#modelMerton.save("./models/MertonModelNewTrans")
#model3NN.save("./models/3NNTrans")
#model2NN.save("./models/2NNTrans")
#modelBSpremium.save("./models/BS/BSpremium")
#modelBS.save("./models/BS/BS")

#modelBS = tf.keras.models.load_model("./models/BS/BS")
# modelBergomi = tf.keras.models.load_model("./models/rBergomiModelTransactionCosts")
# modelHeston = tf.keras.models.load_model("./models/HestonModelTransactionCosts")
#modelMerton = tf.keras.models.load_model("./models/MertonModelNewTrans")

################################################################################
#comparing solo models to joint models

# resultsBergomi = modelBergomi(xtestBerg).numpy().squeeze()
# results3NNBergomi = model3NN(xtestBerg).numpy().squeeze()
# print("Bergomi MC price: ", payoffBerg.mean())
# print("Bergomi NN price: ", resultsBergomi.mean())
# print("Bergomi NN price 3NN: ", results3NNBergomi.mean())
# print("Bergomi squared error: ", ((resultsBergomi-resultsBergomi.mean())**2).mean())
# print("Bergomi squared error 3NN: ", ((results3NNBergomi-results3NNBergomi.mean())**2).mean())

# fig_PnL = plt.figure(dpi= 125, facecolor='w')
# fig_PnL.suptitle("3NN vs rBergomi NN\n", \
#       fontweight="bold")
# ax = fig_PnL.add_subplot()
# ax.set_title("PnL on rBergomi data with transaction costs: eps=0.01", \
#       fontsize=8)
# ax.set_xlabel("PnL")
# ax.set_ylabel("Frequency")
# ax.hist((resultsBergomi, results3NNBergomi), bins=50, \
#           label=[ "rBergomi NN PnL", "3NN PnL"])
# ax.legend()
# plt.show()


# resultsHeston = modelHeston(xtestHest).numpy().squeeze()
# results3NNHeston = model3NN(xtestHest).numpy().squeeze()
# print("Heston MC price: ", payoffHest.mean())
# print("Heston NN price: ", resultsHeston.mean())
# print("Heston NN price 3NN: ", results3NNHeston.mean())
# print("Heston squared error: ", ((resultsHeston-resultsHeston.mean())**2).mean())
# print("Heston squared error 3NN: ", ((results3NNHeston-results3NNHeston.mean())**2).mean())


# fig_PnL = plt.figure(dpi= 125, facecolor='w')
# fig_PnL.suptitle("3NN vs Heston NN\n", \
#       fontweight="bold")
# ax = fig_PnL.add_subplot()
# ax.set_title("PnL on Heston data with transaction costs: eps=0.01", \
#       fontsize=8)
# ax.set_xlabel("PnL")
# ax.set_ylabel("Frequency")
# ax.hist((resultsHeston, results3NNHeston), bins=50, \
#           label=[ "Heston NN PnL", "3NN PnL"])
# ax.legend()
# plt.show()


# resultsMerton = modelMerton(xtestMert).numpy().squeeze()
# results3NNMerton = model3NN(xtestMert).numpy().squeeze()
# print("Merton MC price: ", payoffMert.mean())
# print("Merton NN price: ", resultsMerton.mean())
# print("Merton NN price 3NN: ", results3NNMerton.mean())
# print("Merton squared error: ", ((resultsMerton-resultsMerton.mean())**2).mean())
# print("Merton squared error 3NN: ", ((results3NNMerton-results3NNMerton.mean())**2).mean())


# fig_PnL = plt.figure(dpi= 125, facecolor='w')
# fig_PnL.suptitle("3NN vs Merton NN\n", \
#       fontweight="bold")
# ax = fig_PnL.add_subplot()
# ax.set_title("PnL on Merton data with transaction costs: eps=0.01", \
#       fontsize=8)
# ax.set_xlabel("PnL")
# ax.set_ylabel("Frequency")
# ax.hist((resultsMerton, results3NNMerton), bins=50, \
#           label=[ "Merton NN PnL", "3NN PnL"])
# ax.legend()
# plt.show()

################################################################################
#comparing different hedges in the bs model

# deltaBS = assetsBS.deltaHedgeCall(Stest[0], strike)
# testPayoff = callPayoff(strike, Stest[0])
# PnLDeltaHedge = assetsBS.PnL(Stest[0], testPayoff, deltaBS, eps)

# resultsBS = modelBS(xtestBS).numpy().squeeze()

# resultsBSpremium, premium = modelBSpremium(xtestBS)
# resultsBSpremium = resultsBSpremium.numpy().squeeze()
# premium = premium.numpy().squeeze().mean()
# premiumAverge = resultsBSpremium.mean()


# print("------------------------")
# print("BS analytics price: ", priceBS)
# print("BS MC price: ", payoffBS.mean())
# print("Delta Hedge price: ", PnLDeltaHedge.mean())

# print("NN premium: ", premium)
# print("NN premium avergae", premiumAverge)
# print("BS NN price: ", resultsBS.mean())

# print("----------------")
# print("Delta Hedge squared error: ", ((PnLDeltaHedge-PnLDeltaHedge.mean())**2).mean())
# print("BSpremium NN squared error: ", ((resultsBSpremium-premiumAverge)**2).mean())
# print("BS NN squared error: ", ((resultsBS-resultsBS.mean())**2).mean())



# fig_PnL = plt.figure(dpi= 125, facecolor='w')
# fig_PnL.suptitle("Deep Hedging PnL BlackScholes \n", \
#       fontweight="bold")
# ax = fig_PnL.add_subplot()
# ax.set_title("No transaction costs", \
#       fontsize=8)
# ax.set_xlabel("PnL")
# ax.set_ylabel("Frequency")
# ax.hist((resultsBSpremium, resultsBS-resultsBS.mean(), PnLDeltaHedge+priceBS), bins=30, \
#           label=[ "NN premium", "NN simple", "Delta Hedge"])
# ax.legend()
# plt.savefig("./graphs/BlackScholes/PnL2")
# plt.show()

################################################################################
#track trading volume

# resultsBergomi,volumeBergomi = modelBergomi(xtestBerg)
# resultsBergomi = resultsBergomi.numpy().squeeze()
# volumeBergomi = volumeBergomi.numpy().squeeze().mean()
# print("Bergomi NN price: ", resultsBergomi.mean())
# print("Bergomi squared error: ", ((resultsBergomi-resultsBergomi.mean())**2).mean())
# print("Bergomi avg volume: ", volumeBergomi)

# resultsHeston,volumeHeston = modelHeston(xtestHest)
# resultsHeston = resultsHeston.numpy().squeeze()
# volumeHeston = volumeHeston.numpy().squeeze().mean()
# print("Heston NN price: ", resultsHeston.mean())
# print("Heston squared error: ", ((resultsHeston-resultsHeston.mean())**2).mean())
# print("Heston avg volume: ", volumeHeston)

# resultsMerton,volumeMerton = modelMerton(xtestMert)
# resultsMerton = resultsMerton.numpy().squeeze()
# volumeMerton = volumeMerton.numpy().squeeze().mean()
# print("Merton NN price: ", resultsMerton.mean())
# print("Merton squared error: ", ((resultsMerton-resultsMerton.mean())**2).mean())
# print("Merton avg volume: ", volumeMerton)

results3NNBergomi = model3NN(xtestBerg)[1].numpy().squeeze().mean()
results3NNBergomieps = model3NNeps(xtestBerg)[1].numpy().squeeze().mean()

results3NNHeston = model3NN(xtestHest)[1].numpy().squeeze().mean()
results3NNHestoneps = model3NNeps(xtestHest)[1].numpy().squeeze().mean()

results3NNMerton = model3NN(xtestMert)[1].numpy().squeeze().mean()
results3NNMertoneps = model3NNeps(xtestMert)[1].numpy().squeeze().mean()

print("Berg 3NN trading volume: ", results3NNBergomi)
print("Bergeps 3NN trading volume: ", results3NNBergomieps)

print("Hest 3NN trading volume: ", results3NNHeston)
print("Hesteps 3NN trading volume: ", results3NNHestoneps)

print("Mert 3NN trading volume: ", results3NNMerton)
print("Merteps 3NN trading volume: ", results3NNMertoneps)

################################################################################
#results of the 2NN

# resultsBergomi = modelBergomi(xtestBerg).numpy().squeeze()
# results2NNBergomi = model2NN(xtestBerg).numpy().squeeze()
# print("Bergomi MC price: ", payoffBerg.mean())
# print("Bergomi NN price: ", resultsBergomi.mean())
# print("Bergomi NN price 2NN: ", results2NNBergomi.mean())
# print("Bergomi squared error: ", ((resultsBergomi-resultsBergomi.mean())**2).mean())
# print("Bergomi squared error 2NN: ", ((results2NNBergomi-results2NNBergomi.mean())**2).mean())

# fig_PnL = plt.figure(dpi= 125, facecolor='w')
# fig_PnL.suptitle("2NN vs rBergomi NN\n", \
#       fontweight="bold")
# ax = fig_PnL.add_subplot()
# ax.set_title("PnL on rBergomi data with transaction costs: eps=0.01", \
#       fontsize=8)
# ax.set_xlabel("PnL")
# ax.set_ylabel("Frequency")
# ax.hist((resultsBergomi, results2NNBergomi), bins=50, \
#           label=[ "rBergomi NN PnL", "2NN PnL"])
# ax.legend()
# plt.show()

# resultsHeston = modelHeston(xtestHest).numpy().squeeze()
# results2NNHeston = model2NN(xtestHest).numpy().squeeze()
# print("Heston MC price: ", payoffHest.mean())
# print("Heston NN price: ", resultsHeston.mean())
# print("Heston NN price 2NN: ", results2NNHeston.mean())
# print("Heston squared error: ", ((resultsHeston-resultsHeston.mean())**2).mean())
# print("Heston squared error 2NN: ", ((results2NNHeston-results2NNHeston.mean())**2).mean())

# fig_PnL = plt.figure(dpi= 125, facecolor='w')
# fig_PnL.suptitle("2NN vs Heston NN\n", \
#       fontweight="bold")
# ax = fig_PnL.add_subplot()
# ax.set_title("PnL on Heston data with transaction costs: eps=0.01", \
#       fontsize=8)
# ax.set_xlabel("PnL")
# ax.set_ylabel("Frequency")
# ax.hist((resultsHeston, results2NNHeston), bins=50, \
#           label=[ "Heston NN PnL", "2NN PnL"])
# ax.legend()
# plt.show()







