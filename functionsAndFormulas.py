import numpy as np
from sklearn import model_selection


"""
 This file contains different useful functions to manipulate data that we use 
 """

def createInput(priceAsset, priceSwap, I1, I2, payoff, N):
    xAll = []
    for i in range(N+1):
      xAll += [np.hstack((priceAsset[i,:,None], priceSwap[i,:,None]))]
      if i != N:
        xAll += [np.hstack((I1[i,:,None], I2[i,:,None]))]
    xAll += [payoff[:,None]]
    return xAll


#paths must be of format: each row represents a different path;
#each column respresents a different time step
#only works for one asset
def increments(paths):
    return paths[:,1:]-paths[:,0:-1]


def listInput(increments,N):
    return np.hsplit(increments, N)


def trainTestSplit(data=None, testSize=None):
    xtrain = []
    xtest = []
    for x in data:
        tmp_xtrain, tmp_xtest = model_selection.train_test_split(
            x, test_size=testSize, shuffle=False)
        xtrain += [tmp_xtrain]
        xtest += [tmp_xtest]
    return xtrain, xtest









