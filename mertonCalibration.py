import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from rbergomi.rbergomi import rBergomi
import logging
from py_vollib.black_scholes.implied_volatility import implied_volatility


"""
code inspired by
https://www.codearmo.com/python-tutorial/merton-jump-diffusion-model-python
note that the code in this link is not correct and is fixed below

The two issues are: 
1. In the simulation of paths the author multiplies a normal distribution times the 
outcome of the poisson process however n*Z does not have the same distribution as the sum
Z_1+...+Z_n which would be the correct term
2. The pricing formula uses m when it should use m* = exp(m+0.5*v**2), 
which would just be confusing notation, but the author does not transform this back into
the model parameter: m = ln(m*)-0.5*v**2
"""


def merton_jump_paths(S, T, r, sigma,  lam, m, v, steps, Npaths):
    size=(steps,Npaths)
    dt = T/steps 
    poi_rv = np.multiply(np.random.poisson( lam*dt, size=size),
                         np.random.normal(m,v, size=size)).cumsum(axis=0)
    geo = np.cumsum(((r -  sigma**2/2 -lam*(m  + v**2*0.5))*dt +\
                              sigma*np.sqrt(dt) * \
                              np.random.normal(size=size)), axis=0)
    
    return np.exp(geo+poi_rv)*S

S0 = 1 # current stock price
K = 1
T = 30/365 # time to maturity
r = 0.0 # risk free rate

#Bergomi params
a = -0.4        #H-0.5 (Hurst param)
H = 0.1
xi = 0.235**2   #starting variance
eta = 1.9
rho = -0.9 

N = norm.cdf

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)    

def rBergomi_pricer(H, eta, rho, v0, tau, K, S0, MC_samples=40000):
    """Computes European Call price under rBergomi dynamics with MC sampling.
    
    """
    try:
        rB = rBergomi(n=365, N=MC_samples, T=tau, a=H-0.5)
        dW1, dW2 = rB.dW1(), rB.dW2()
        Y = rB.Y(dW1)
        dB = rB.dB(dW1, dW2, rho)
        xi = v0
        V = rB.V(Y, xi, eta)
        S = rB.S(V, dB)
        ST = S[:, -1]
        price = np.mean(np.maximum(ST-K, 0))
    except:
        return np.nan, np.nan
    
    # check numerical stability
    if price <= 0 or price + K < S0:
        iv = np.nan
        logging.debug("NumStabProblem: Price {}. Intrinsic {}. Time {}. Strike {}.".format(price, S0-K, tau, K))
    else:
        logging.debug("Success: Price {} > intrinsic {}".format(price, S0-K))
        iv = implied_volatility(price, S0, K, tau, 0, 'c')
    return price    

def merton_jump_call(S, K, T, r, sigma, m , v, lam):
    p = 0
    for k in range(40):
        r_k = r - lam*(m-1) + (k*np.log(m) ) / T
        sigma_k = np.sqrt( sigma**2 + (k* v** 2) / T)
        k_fact = np.math.factorial(k)
        p += (np.exp(-m*lam*T) * (m*lam*T)**k / (k_fact))  * BS_CALL(S, K, T, r_k, sigma_k)  
    return p

def merton_jump_put(S, K, T, r, sigma, m , v, lam):
    p = 0 # price of option
    for k in range(40):
        r_k = r - lam*(m-1) + (k*np.log(m) ) / T
        sigma_k = np.sqrt( sigma**2 + (k* v** 2) / T)
        k_fact = np.math.factorial(k) # 
        p += (np.exp(-m*lam*T) * (m*lam*T)**k / (k_fact)) \
                    * BS_PUT(S, K, T, r_k, sigma_k)
    return p 

def optimal_params(x, mkt_prices, strikes):
    candidate_prices = merton_jump_call(S0, strikes, T, r,
                                        sigma=x[0], m= x[1] ,
                                        v=x[2],lam= x[3])
    return np.linalg.norm(mkt_prices - candidate_prices, 2)

strikes = np.linspace(0.9, 1.1, 113)
prices = [rBergomi_pricer(H, eta, rho, xi, T, strike, S0) for strike in strikes]

x0 = [0.2, 1.0, 0.1, 1] #sigma , m(mu), v(delta), lambda
bounds = ((0.01, np.inf) , (0.01, 3), (1e-5, np.inf) , (0, 20)) #bounds as described above

res = minimize(optimal_params, method='SLSQP',  x0=x0, args=(prices, strikes),
                  bounds = bounds, tol=1e-20, 
                  options={"maxiter":1000})
sigt = res.x[0]
mt = res.x[1]
vt = res.x[2]
lamt = res.x[3]

print('Calibrated Volatlity = ', sigt)
print('Calibrated Jump Std = ', vt)
print('Calibrated Jump Mean = ', np.log(mt)-0.5*vt**2)
print('Calibrated intensity = ', lamt)































