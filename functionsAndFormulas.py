import numpy as np

#paths must be of format: each row represents a different path;
#each column respresents a different time step
def call(K,paths):
    return (paths[:,-1]-K).clip(min=0)
   

def VaR(alpha, loss):
    return np.quantile(loss, 1-alpha)


def cvar(alpha, loss):
    VaR = np.quantile(loss, 1-alpha)
    return 1/(1-alpha)*loss[loss<=VaR].mean()


#paths must be of format: each row represents a different path;
#each column respresents a different time step
#only works for one asset
def increments(paths):
    return paths[:,1:]-paths[:,0:-1]

#multiple assets of shape (different paths, different timesteps, different assets)
# i.e. same as 1 asset with a third dimension for different assets
def incrementsMulti(paths):
    pass

def listInput(increments,N):
    return np.hsplit(increments, N)













