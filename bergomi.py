import numpy as np
"""
this implementation leans on and adats https://github.com/ryanmccrickerd/rough_bergomi
"""

class Bergomi:
    def __init__(self, T, N, a, rho, eta, xi, S0, Tf=0):
        
        self.T=T  #maturity
        self.N=N  #number of time steps
        self.a=a  #alpha oder a, param of volteraa process (Hurst+0.5)
        self.rho=rho  #corr
        self.eta=eta  #this is v in the original heding paper
        self.xi=xi    #xi0 in og hedg paper
        self.S0=S0    #starting price
        #maturity of forward variance
        if Tf > 0:
            self.Tf = Tf
        else:
            self.Tf = T

    def generatePaths(self, M):
 
        #create stochastic drivers
        dW1, dW2 = self.dW(M, self.T, self.N)
        dB = self.rho * dW1[:,:,0] + np.sqrt(1 - self.rho**2) * dW2

        #create first part of volterra process
        Y = self.Y(dW1, M)
        
        #create the variance process
        t = np.linspace(0, self.T, 1 + self.N)[np.newaxis,:] # Time grid
        volt = self.eta * Y
        V = self.xi * np.exp(volt - 0.5 * self.eta**2 * t**(2 * self.a + 1))
        
        
        #create the asset price process
        dt=self.T / self.N
        increments = np.sqrt(V[:,:-1]) * dB - 0.5 * V[:,:-1] * dt

        integral = np.cumsum(increments, axis = 1)

        S = np.zeros_like(V)
        S[:,0] = self.S0
        S[:,1:] = self.S0 * np.exp(integral)
        
        
        #create the swap price process
        dw = dW1[:,:,0]
        dtMatrix = np.full_like(dw, dt)
        times = (self.Tf-np.linspace(0, self.T-self.T/self.N, self.N))**self.a
        incr1 = np.sqrt(2*self.a+1)*self.eta*dw*times
        incr2 = -0.5*(2*self.a+1)*self.eta**2*dtMatrix*times

        integralSwap = np.cumsum(incr1+incr2, axis=1)
        
        Swap = np.zeros_like(V)
        Swap[:,0] = self.xi
        Swap[:,1:] = self.xi * np.exp(integralSwap)
        
        return S, V, Swap
    
    def modelInput(self, M):
        
        S, v, Swap = self.generatePaths(M)
        
        lnS = np.log(S)
        
        
        return S, Swap, lnS, v
        
    def Y(self, dW, M):
        """
        Constructs Volterra process from appropriately
        correlated 2d Brownian increments.
        """
        Y1 = np.zeros((M, 1 + self.N)) # Exact integrals
        Y2 = np.zeros((M, 1 + self.N)) # Riemann sums

        # Construct Y1 through exact integral
        for i in np.arange(1, 1 + self.N, 1):
            Y1[:,i] = dW[:,i-1,1] # Assumes kappa = 1

        # Construct arrays for convolution
        G = np.zeros(1 + self.N) # Gamma
        for k in np.arange(2, 1 + self.N, 1):
            G[k] = g(b(k, self.a)/365.0, self.a)

        X = dW[:,:,0] # Xi

        # Initialise convolution result, GX
        GX = np.zeros((M, len(X[0,:]) + len(G) - 1))

        # Compute convolution, FFT not used for small n
        # Possible to compute for all paths in C-layer?
        for i in range(M):
            GX[i,:] = np.convolve(G, X[i,:])

        # Extract appropriate part of convolution
        Y2 = GX[:,:1 + self.N]

        # Finally contruct and return full process
        Y = np.sqrt(2 * self.a + 1) * (Y1 + Y2)
        return Y

    def dW(self, M, T, N):
        
        rng = np.random.multivariate_normal
        a = self.a
        e = np.array([0,0])
        dt = T/N
        
        cov = np.array([[0.,0.],[0.,0.]])
        cov[0,0] = 1./365.0
        cov[0,1] = 1./((1.*a+1) * 365.0**(1.*a+1))
        cov[1,1] = 1./((2.*a+1) * 365.0**(2.*a+1))
        cov[1,0] = cov[0,1]         

        return rng(e, cov, (M, N)), (np.random.randn(M, N) * np.sqrt(dt))

    
def g(x, a):
    return x**a
        
def b(k, a):
    return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)  
        
        
        
    