import numpy as np

def callPayoff(K, paths):
    return np.maximum(paths[:,-1] - K, 0.0)