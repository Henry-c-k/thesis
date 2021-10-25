import numpy as np

def call(K,paths):
    return (paths[:,-1]-K).clip(min=0)
   

def VaR(alpha, loss):
    return np.quantile(loss, 1-alpha)


def cvar(alpha, loss):
    VaR = np.quantile(loss, 1-alpha)
    return 1/(1-alpha)*loss[loss<=VaR].mean()