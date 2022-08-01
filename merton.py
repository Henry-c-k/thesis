import numpy as np

def addNormals(matrix, mu, delta):
    """
    fill every entry of a matrix of integers with sum of that many normal distributions
    """
    
    (m, n) = matrix.shape
    
    x = np.zeros(matrix.shape)
    
    for i in range(m):
        for j in range(n):
            if matrix[i,j] != 0:
                new = np.random.normal(mu, delta, int(matrix[i,j]))
                x[i,j] = new.sum()
    
    return x


class Merton:
    
    def __init__(self, T, N, S0, lam, mu, delta, sigma, Tf=0, r=0):
        self.T = T
        self.N = N
        self.S0 = S0
        self.lam = lam
        self.mu = mu
        self.delta = delta  
        self.sigma = sigma
        self.r = r
        if Tf>0:
            self.Tf = Tf
        else:
            self.Tf = T
        
    def generateReturns(self, numberOfSimulations):
        """
        calculation is the same as in BS case with the added poisson sum of 
        normal variables
        """
        
        dt = self.T/self.N
        size = (numberOfSimulations, self.N)
        k = np.exp(self.mu + 0.5 * self.delta**2)-1

        #for clarity of computations
        factor = np.exp((self.r - 0.5 * self.sigma**2 - self.lam*k) * dt)
        
        #stochastic driver
        Z = np.random.standard_normal(size)
        poiss = np.random.poisson(self.lam*dt, size=size)
        comPoiss = addNormals(poiss, self.mu, self.delta)

        return factor*np.exp(self.sigma*np.sqrt(dt)*Z + comPoiss)
        
        
    def generatePaths(self, numberOfSimulations):
        
        return np.cumprod(np.hstack((np.full([numberOfSimulations,1],self.S0),\
                   self.generateReturns(numberOfSimulations))),axis=1)
            
    def getInformation(self, S):
        return np.log(S), np.full(S.shape, self.sigma**2)
    
    def getSwap(self, v):
        """
        sigma not stochastic so swap is just the v*maturity
        note that v is not the variance of the price process
        """ 
        
        return np.full_like(v, (self.sigma**2+self.lam*(self.mu**2+self.delta**2))*self.Tf ) 
            
            
            
            
            
            
        
        