import numpy as np
from scipy.stats import norm

class BlackScholes:
    
    def __init__(self, finalTime, sigma, S0, numberOfTimes,
                 r = 0, Tf=0):
        
        self.S0 = S0
        self.finalTime = finalTime
        self.sigma = sigma
        self.r = r
        self.numberOfTimes = numberOfTimes
        if Tf > 0:
            self.Tf = Tf
        else:
            self.Tf = finalTime
        
        
    def generateReturns(self, numberOfSimulations):
        
        lenghthOfIntervals = self.finalTime / self.numberOfTimes 
        
        blackScholesReturns = []
        
        #for clarity of computations
        firstPart = np.exp((self.r - 0.5 * self.sigma**2) * lenghthOfIntervals)
        
        #stochastic driver
        Z = np.random.standard_normal([numberOfSimulations,self.numberOfTimes])
        
        blackScholesReturns = firstPart*np.exp(self.sigma*np.sqrt(lenghthOfIntervals)*Z)
    
        return blackScholesReturns
        
    def generatePaths(self, numberOfSimulations):
        
        return np.cumprod(np.hstack((np.full([numberOfSimulations,1],self.S0),\
                   self.generateReturns(numberOfSimulations))),axis=1)
            
            
    def deltaHedgeCall(self, paths, K):
        
        times = np.arange(0, self.finalTime, self.finalTime/self.numberOfTimes)
        
        ttm = self.finalTime - times
        
        dplus =(np.log(paths[:,0:-1]/K)+(self.r+0.5*self.sigma**2)*ttm)/(self.sigma*np.sqrt(ttm))
        
        return norm.cdf(dplus)
      
    def increase(self, paths):
        return paths[:,1:,]-paths[:,0:-1]
        
    def tradingReturn(self, hedge, paths):
        return np.sum(hedge*self.increase(paths),axis=1)
    
        
    def blackScholesPriceCall(self, strike):
    
        d1 = (np.log(self.S0 / strike) + (self.r + 0.5 * self.sigma ** 2)\
              * self.finalTime)/(self.sigma * np.sqrt(self.finalTime))
            
        d2 = (np.log(self.S0 / strike) + (self.r - 0.5 * self.sigma ** 2)\
              * self.finalTime)/(self.sigma * np.sqrt(self.finalTime))
        
        callPrice = (self.S0 * norm.cdf(d1) - strike * np.exp(-self.r * self.finalTime)\
                     * norm.cdf(d2))
      
        return callPrice
    
    def PnL(self, paths, payoff, deltaHedge, eps):
        
        """
        Add up the cost of purchasing/changing position in assets, the associated transaction cost
        and the payoff and set it against the money earning from liquidating the final position, 
        all from the viewpoint of the bank. constant transaction costs not usable right now
        """
        costType = "proportional"
        wealth = np.multiply(paths[:,0], -deltaHedge[:,0])
        
        if costType == "proportional":
            wealth = wealth - np.abs(deltaHedge[:,0])*paths[:,0]*eps 
        elif costType == "constant":
            wealth = wealth - eps
        
        for t in range(1, self.numberOfTimes):
            
            wealth = wealth + np.multiply(paths[:,t], -deltaHedge[:,t] + deltaHedge[:,t-1])
            
            if costType == "proportional":
                wealth = wealth - np.abs(deltaHedge[:,t]-deltaHedge[:,t-1])*paths[:,t]*eps 
            elif costType == "constant":
                wealth = wealth - eps
        
        wealth = wealth + np.multiply(paths[:,self.numberOfTimes], deltaHedge[:,self.numberOfTimes-1]) - payoff
        
        if costType == "proportional":
            wealth = wealth - np.abs(deltaHedge[:,self.numberOfTimes-1])*paths[:,self.numberOfTimes]*eps
        elif costType == "constant":
            wealth = wealth - eps
        
        return wealth
        
        
    
    
    
        
        
        
        
        
        