import numpy as np
import tensorflow.keras.backend as K

def VaR(alpha, loss):
    return np.quantile(loss, 1-alpha)

def cvar(alpha, loss):
    VaR = np.quantile(loss, 1-alpha)
    return 1/(1-alpha)*loss[loss<=VaR].mean()

def entropyLoss(x=None, lam=None):
    return (1/lam)*K.log(K.mean(K.exp(-lam*x)))

def MSE(x):
    return K.mean(K.square(x))

def CVaR(x = None, w = None, loss_param = None):
    alpha = loss_param
    # Expected shortfall risk measure
    return K.mean(w + (K.maximum(-x-w,0)/(1.0-alpha)))