import numpy as np
import QuantLib as ql

class Heston:
    
    def __init__(self, T, sigma, kappa, theta, corr, S0, N, v0, Tf=0, r=0):#r not 0 is not implemented
        
        self.S0 = S0
        self.T = T
        self.sigma = sigma #eps in dem einen paper
        self.N = N
        self.corr = corr
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        if Tf > 0:
            self.Tf = Tf
        else:
            self.Tf = T
        
# sigma is volatility of volatility 
# theta long term price variance
# k drift of vola
# corr of B.M.
#M number simulations
#N number of time steps
#T maturiy

    def generateHestonPathsEuler(self, M):
        
        """
        simple euler scheme for sampling (with small fix for negative variance)
        this was not used for any experiments other than comparing its simulation
        accuracy to that of the scheme below
        """
        
        dt = self.T / self.N
        cov = np.array([[1,self.corr],
                       [self.corr,1]])
    
        # arrays for storing prices and variances
        S = np.full(shape=(self.N+1,M), fill_value=self.S0)
        v = np.full(shape=(self.N+1,M), fill_value=self.v0)
    
        # sampling correlated brownian motions
        Z = np.random.multivariate_normal(np.array([0,0]), cov, (self.N,M))
    
        for i in range(1,self.N+1):
            S[i] = S[i-1] * np.exp( ( - 0.5*v[i-1])*dt + np.sqrt(v[i-1] * dt) * Z[i-1,:,0] )
            v[i] = np.maximum(v[i-1] + self.kappa*(self.theta-v[i-1])*dt + self.sigma*np.sqrt(v[i-1]*dt)*Z[i-1,:,1],0)
       
        return S.transpose(), v.transpose()

    def generateHestonPaths(self, M, gamma1=1.0, gamma2=0.0):
        """
        broadie kaya scheme with simplified integral v(u)du
        this can directly be used as information process
        """
        lnS = np.full(shape=(self.N+1,M), fill_value=np.log(self.S0))
        v = np.full(shape=(self.N+1,M), fill_value=self.v0)
               
        df = 4*self.theta*self.kappa/(self.sigma)**2 #degrees_of_freedom
        dt = self.T / self.N
        
        for i in range(1, self.N+1):
            #noncentrality parameter
            nonc1 = 4*self.kappa*np.exp(-self.kappa*dt)*v[i-1,:]
            nonc2 = (self.sigma)**2*(1-np.exp(-self.kappa*dt))
            nonc = nonc1/nonc2
            vorfaktor = self.sigma**2*(1-np.exp(-self.kappa*dt))/(4*self.kappa)
            v[i,:] = vorfaktor*np.random.noncentral_chisquare(df, nonc, M)
            
            #approximate the otherwise tricky integral 
            int_v_du = dt*(gamma1*v[i-1,:]+gamma2*v[i,:])
            
            sum1 = self.corr/self.sigma*(v[i,:]-v[i-1,:]-self.kappa*self.theta*dt)
            sum2 = (self.kappa*self.corr/self.sigma-0.5)*int_v_du
            sum3 = np.sqrt(1-self.corr**2)*np.random.normal(0, np.sqrt(int_v_du), M)
            
            lnS[i,:] = lnS[i-1,:]+sum1+sum2+sum3
                    
        return lnS.transpose(), v.transpose()
    
    def generateAssetPaths(self, lnS, v, M, gamma1=1.0, gamma2=0.0):
        """
        simple approximation to get the price of the variance swap
        name is misleading this doesnt generate anything it just calculates the 
        assets prices from the information process
        """
        dt = self.T / self.N
    
        swap = np.full(shape=(M, self.N+1), fill_value=self.v0)
        
        #integral is zero for the starting value
        integral = np.zeros(v[:,0].shape)
        swap[:,0] = (v[:,0]-self.theta)*(1-np.exp(-self.kappa*self.Tf))/self.kappa+self.theta*self.Tf
        
        for i in range(1, self.N+1):
        
            integral = integral + dt*(gamma2*v[:,i-1]+gamma1*v[:,i])
            L = (v[:,i]-self.theta)*(1-np.exp(-self.kappa*(self.Tf-dt*i)))/self.kappa+self.theta*(self.Tf-dt*i)
            
            swap[:,i] = integral + L

        return np.exp(lnS), swap
    
    
    def increase(self, paths):
        return paths[:,1:,]-paths[:,0:-1]
    

    def hestonPriceCall(self, strike):
        """code pieced together from quantlib tutorial and quantlib doc:
        http://gouthamanbalaraman.com/blog/valuing-european-option-heston-model-quantLib.html
        """
        
        risk_free_rate = 0.0
        dividend_rate =  0.0

        maturity_date = ql.Date.todaysDate() + self.N
        calculation_date = ql.Date.todaysDate()

        day_count = ql.Actual365Fixed()
        ql.Settings.instance().evaluationDate = calculation_date
        option_type = ql.Option.Call

        # construct the European Option
        payoff = ql.PlainVanillaPayoff(option_type, strike)
        exercise = ql.EuropeanExercise(maturity_date)
        european_option = ql.VanillaOption(payoff, exercise)

        #create heston process
        spot_handle = ql.QuoteHandle(
            ql.SimpleQuote(self.S0)
        )
        flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date, risk_free_rate, day_count)
        )
        dividend_yield = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date, dividend_rate, day_count)
        )
        heston_process = ql.HestonProcess(flat_ts,
                                          dividend_yield,
                                          spot_handle,
                                          self.v0,
                                          self.kappa,
                                          self.theta,
                                          self.sigma,
                                          self.corr)

        engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process),0.01, 1000)
        european_option.setPricingEngine(engine)
        h_price = european_option.NPV()
        
        return  h_price   